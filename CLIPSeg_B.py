# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 15:53:07 2025

@author: Julia Dietlmeier <julia.dietlmeier@insight-centre.org>
"""

from typing import Optional

from torch import nn
import torch
from transformers import CLIPSegForImageSegmentation
from transformers.models.clipseg.modeling_clipseg import CLIPSegImageSegmentationOutput
import math
import torchvision.transforms as transforms

resize_transform1 = transforms.Resize((352, 352))
resize_transform2 = transforms.Resize((112, 112))

#https://github.com/changzy00/pytorch-attention/blob/master/attention_mechanisms/eca.py  
class ECALayer(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        y = self.avgpool(x)       
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class CLIPSeg(nn.Module):
    r"""CLIPSeg Official implementation from HuggingFace.

    Args:
        clipseg_hf_api (str): HuggingFace api to import the CLIPSeg implementation; Eg:'CIDAS/clipseg-rd64-refined'
        freeze_encoder (bool): Whether or not to freeze the encoders of pretrained CLIPSeg; Default is False.
        freeze_decoder (bool): Whether or not to freeze the decoder of pretrained CLIPSeg; Default is False.
    """
    def __init__(
        self,
        clipseg_hf_api: str,
        freeze_encoder: bool = False,
        freeze_decoder: bool = False,
    ) -> None:
        super().__init__()

        self.clipseg = CLIPSegForImageSegmentation.from_pretrained(clipseg_hf_api,
                                                                   output_attentions= True,
                                                                   output_hidden_states= True)

        self.clipseg.clip.requires_grad_(not freeze_encoder)
        self.clipseg.decoder.requires_grad_(not freeze_decoder)
        self.clipseg.output = CLIPSegImageSegmentationOutput
        
        tp_kernels = (16 // 4, 16 // 4)
        reduce_dim=128
        n_heads=4
        self.extract_layers = (3,6,9)
        depth = len((3,6,9))
         
        self.film_mul = nn.Linear(512, reduce_dim)
        self.film_add = nn.Linear(512, reduce_dim)

        self.reduce = nn.Linear(768, reduce_dim)
        self.reduce2 = nn.Linear(485, 484)

        self.trans_conv = nn.ConvTranspose2d(reduce_dim, 1, 16, stride=16)
        
        self.trans_conv1 = nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(reduce_dim, reduce_dim // 2, kernel_size=tp_kernels[0], stride=tp_kernels[0]),
                nn.ReLU(),
                nn.ConvTranspose2d(reduce_dim // 2, 1, kernel_size=tp_kernels[1], stride=tp_kernels[1]),               
            )
        
        self.upsample_proj = nn.Conv2d(reduce_dim, 1, kernel_size=1)
        self.convlast=nn.Conv2d(2,1,1)
        self.decoder2 = UNet_D(1, nn.BatchNorm2d)
        
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d_model=reduce_dim, nhead=n_heads) for _ in range(len(self.extract_layers))])
        self.reduces = nn.ModuleList([nn.Linear(768, reduce_dim) for _ in range(depth)])
        
    def forward(
        self, 
        input_ids:torch.Tensor, 
        pixel_values:torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None, 
        **kwargs
    ) -> torch.Tensor:
        
        B, C, H, W = pixel_values.shape       
        
        outputs_2 = self.clipseg.output(self.clipseg(input_ids=input_ids,pixel_values=pixel_values,attention_mask=attention_mask))

#------------------------------------------------------------------------------
#-------DATA ADAPTER ----------------------------------------------------------        
              
        conditional_embeddings=outputs_2[1]

        vision_model_output=outputs_2[3]
        
        act0=vision_model_output[0]
        act1=vision_model_output[1]
        attention_mask=act1

        decoder_output=outputs_2[4]
        
        csa=conditional_embeddings.unsqueeze(1)

        a = None
        _activations=act0
        for i, (activation, block, reduce) in enumerate(zip(_activations, self.blocks, self.reduces)):
            
            if a is not None:
                a = reduce(activation) + a
            else:
                a = reduce(activation)
                a = self.film_mul(csa) * a + self.film_add(csa)

            a = block(a)

        ar = a.permute(0,2,1)
        size = int(math.sqrt(ar.shape[2]))
      
        a = self.reduce2(ar).view(pixel_values.shape[0], self.reduce2(ar).shape[1], size, size)
            
#------------------------------------------------------------------------------       
        
        decoder_outputs = self.decoder2(pixel_values, a, decoder_output[0].unsqueeze(1))

        return self.convlast(torch.cat([decoder_output[0].unsqueeze(1),decoder_outputs],dim=1))


class UNet_D(nn.Module):
    def __init__(self, num_classes, BatchNorm):
        
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        
        self.conv11 = nn.Conv2d(3, 64, 3, padding=1, bias=True)
        self.conv12 = nn.Conv2d(64, 64, 3, padding=1, bias=True)
        self.gn1 = nn.GroupNorm(4,64)
        self.bn1 = BatchNorm(64)
        self.maxpool = nn.MaxPool2d(2)
        
        self.conv22 = nn.Conv2d(64, 128, 3,  padding=1,bias=True)
        self.conv222 = nn.Conv2d(128, 128, 3,  padding=1,bias=True)
        self.bn2 = BatchNorm(128)
        
        self.conv33 = nn.Conv2d(128, 256, 3, padding=1,bias=True)
        self.conv333 = nn.Conv2d(256, 256, 3, padding=1,bias=True)
        self.bn3 = BatchNorm(256)
        
        self.conv44 = nn.Conv2d(256, 512, 3,  padding=1,bias=True)
        self.conv444 = nn.Conv2d(512, 512, 3, padding=1,bias=True)
        self.bn4 = BatchNorm(512)
        
        self.upconv41 = nn.Conv2d(640,1240,1)
        self.upconv4 = nn.ConvTranspose2d(1240, 546, kernel_size=2, stride=2)
        self.convup4 = nn.Conv2d(546, 546, 3, padding=1,bias=True)
        self.upbn4 = BatchNorm(546)
        
        self.upconv3 = nn.ConvTranspose2d(802, 182, kernel_size=2, stride=2)
        self.convup3 = nn.Conv2d(182, 182, 3, padding=1,bias=True)
        self.upbn3 = BatchNorm(182)
        
        self.upconv2 = nn.ConvTranspose2d(310, 91, kernel_size=2, stride=2)
        self.convup2 = nn.Conv2d(91, 91, 3, padding=1,bias=True)
        self.upbn2 = BatchNorm(91)
        
        self.upconv1 = nn.ConvTranspose2d(155, 45, kernel_size=2, stride=2)
        self.convup1 = nn.Conv2d(45, 45, 3, padding=1,bias=True)
        self.upbn1 = BatchNorm(45)
        
        self.outconv = nn.Conv2d(45, num_classes, kernel_size=1)
        self.outconv2 = nn.Conv2d(2, num_classes, kernel_size=1)
        
        self.eca4 = ECALayer(1240)
      
        self.Dropout = nn.Dropout(0.1)


    def forward(self, x, low_level_feat, at):
        
        x1 = self.Dropout(self.relu(self.bn1(self.conv12(self.conv11(x)))))
        x1 = self.maxpool(x1)

        x2 = self.Dropout(self.relu(self.bn2(self.conv222(self.conv22(x1)))))
        x2 = self.maxpool(x2)

        x3 = self.Dropout(self.relu(self.bn3(self.conv333(self.conv33(x2)))))
        x3 = self.maxpool(x3)

        x4 = self.Dropout(self.relu(self.bn4(self.conv444(self.conv44(x3)))))
        x4 = self.maxpool(x4)

        xd4=self.relu(self.upbn4(self.convup4(self.convup4(self.upconv4((self.eca4(self.upconv41( torch.cat([low_level_feat,x4],dim=1) ))))))))
        
        xd4 = torch.cat([xd4, x3], dim=1)
        
        xd3=self.relu(self.upbn3(self.convup3(self.convup3((self.upconv3(xd4))))))
        xd3 = torch.cat([xd3, x2], dim=1)
        
        xd2=self.relu(self.upbn2(self.convup2(self.convup2(self.upconv2( xd3 )))))
        xd2 = torch.cat([xd2, x1], dim=1)
        
        xd1=self.relu(self.upbn1(self.convup1(self.convup1(self.upconv1( xd2  )))))
        
        return self.outconv2(torch.cat([at, self.outconv(xd1)],dim=1))

    
