# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 15:53:07 2025

@author: Julia Dietlmeier <julia.dietlmeier@insight-centre.org>
"""
#https://github.com/naamiinepal/medvlsm/blob/main/src/models/biomedclipseg.py
import torch
import torch.nn as nn
from typing import Optional
import open_clip
from open_clip.hf_model import ClsPooler
from transformers import CLIPSegConfig, CLIPSegForImageSegmentation
import torchvision.transforms as transforms
import math
from torch.nn import functional as nnf
from transformers.models.clipseg.modeling_clipseg import CLIPSegImageSegmentationOutput,CLIPSegTextEmbeddings
from transformers import CLIPSegTextConfig

resize_transform1 = transforms.Resize((224, 224))
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

def get_cond_vec(self, conditional, batch_size):
        # compute conditional from a single string
        if conditional is not None and type(conditional) == str:
            cond = self.compute_conditional(conditional)
            cond = cond.repeat(batch_size, 1)

        # compute conditional from string list/tuple
        elif conditional is not None and type(conditional) in {list, tuple} and type(conditional[0]) == str:
            assert len(conditional) == batch_size
            cond = self.compute_conditional(conditional)

        # use conditional directly
        elif conditional is not None and type(conditional) == torch.Tensor and conditional.ndim == 2:
            cond = conditional

        # compute conditional from image
        elif conditional is not None and type(conditional) == torch.Tensor:
            with torch.no_grad():
                cond, _, _ = self.visual_forward(conditional)
        else:
            raise ValueError('invalid conditional')
        return cond   


class BiomedCLIPSeg(nn.Module): # ENSEMBLE Model
    r"""BiomedCLIP Encoder + CLIPSeg Decoder

    Args:
        biomedclip_hf_api (str): HuggingFace api to import the BiomedCLIP implementation; 
            Eg:'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        clipseg_hf_api (str): HuggingFace api to import the CLIPSeg implementation; 
            Eg:'CIDAS/clipseg-rd64-refined'
        freeze_encoder (bool): Whether or not to freeze the encoders of pretrained CLIPSeg; Default is False.
        freeze_decoder (bool): Whether or not to freeze the decoder of pretrained CLIPSeg; Default is False.
        rand_init_decoder (bool): Whether or not to randomly initialize the decoder of pretrained CLIPSeg; Default is True.
    """

    def __init__(
        self,
        biomedclip_hf_api: str,
        clipseg_hf_api: str,
        freeze_encoder: bool = True,
        freeze_decoder: bool = False,
        rand_init_decoder: bool = True,
        n_class=1,
    ):
        super().__init__()
        
        # Encoder from BiomedCLIP
        self.biomedclip = open_clip.create_model(biomedclip_hf_api)
        
        text_config = CLIPSegTextConfig(max_position_embeddings=256)
        
        #self.clip_seg_config = CLIPSegConfig.from_pretrained(clipseg_hf_api)
        self.clip_seg_config = CLIPSegConfig(**text_config.__dict__)

        self.dense_layer=nn.Linear(512,1568)
        
        # Randomly initialize decoder
        self.decoder = CLIPSegForImageSegmentation(self.clip_seg_config).decoder#CNN_Decoder(n_class)
        self.clip_seg = my_CLIPSeg(clipseg_hf_api, False, False)
        
        self.textembed = CLIPSegTextEmbeddings(CLIPSegTextConfig(max_position_embeddings=256))
        
        self.decoder2 = UNet_D(n_class, nn.BatchNorm2d)
              
        reduce_dim=128
        
        self.film_mul = nn.Linear(512, reduce_dim)
        self.film_add = nn.Linear(512, reduce_dim)
        self.reduce_input_ids = nn.Linear(256,77)
        
        self.reduce = nn.Linear(768, reduce_dim)
        self.reduce2 = nn.Linear(197, 196)

        self.trans_conv = nn.ConvTranspose2d(reduce_dim, 1, 16, stride=16)
        
        self.upsample_proj = nn.Conv2d(reduce_dim, 1, kernel_size=1)
        self.convlast=nn.Conv2d(2,1,1)
        
        tp_kernels = (16 // 4, 16 // 4)
    
        self.trans_conv1 = nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(reduce_dim, reduce_dim // 2, kernel_size=tp_kernels[0], stride=tp_kernels[0]),
                nn.ReLU(),
                nn.ConvTranspose2d(reduce_dim // 2, 1, kernel_size=tp_kernels[1], stride=tp_kernels[1]),               
            )

        self.biomedclip.requires_grad_(not freeze_encoder)

    def _forward_vit(self, x, output_hidden_states: bool = True):
        ViT = self.biomedclip.visual.trunk
        x = ViT.patch_embed(x)
        x = ViT._pos_embed(x)
        x = ViT.norm_pre(x)

        hidden_states = []

        for i, block in enumerate(ViT.blocks):
            x = block(x)

            hidden_states.append(x)

        x = ViT.norm(x)

        if ViT.global_pool:
            x = (
                x[:, ViT.num_prefix_tokens :].mean(dim=1)
                if ViT.global_pool == "avg"
                else x[:, 0]
            )
        x = ViT.fc_norm(x)
        x = ViT.head(x)

        # Linear Projection: 768 -> 512
        x = self.biomedclip.visual.head(x)

        if output_hidden_states:
            return x, hidden_states
        else:
            return x

    def _forward_bert(
        self,
        x,
        attention_mask: Optional[torch.LongTensor] = None,
        output_hidden_states: bool = False,
    ):
        bert = self.biomedclip.text

        if attention_mask is None:
            attention_mask = (x != bert.config.pad_token_id).long()

        out = bert.transformer(
            input_ids=x,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )
        pooled_out = bert.pooler(out, attention_mask)
        projected = bert.proj(pooled_out)

        seq_len = out.last_hidden_state.shape[1]
        tokens = (
            out.last_hidden_state[
                :, torch.arange(seq_len) != bert.pooler.cls_token_position, :
            ]
            if type(bert.pooler) == ClsPooler
            else out.last_hidden_state
        )

        if bert.output_tokens:
            return projected, tokens

        if output_hidden_states:
            return projected, out.hidden_states
        else:
            return projected

    def get_conditional_embeddings(
        self,
        batch_size: int,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        # compute conditional embeddings from texts
        if len(input_ids) != batch_size:
            raise ValueError(
                "Make sure to pass as many prompt texts as there are query images"
            )
        conditional_embeddings = self._forward_bert(
            input_ids, attention_mask=attention_mask, output_hidden_states=False
        )
        return conditional_embeddings

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        
        
        
        # step 1: forward the query images through the frozen CLIP vision encoder
        with torch.inference_mode():
            pooled_output, hidden_states = self._forward_vit(
                pixel_values, output_hidden_states=True
            )
            
            #print('ViT pooled output shape = ',pooled_output.shape)
            # we add +1 here as the hidden states also include the initial embeddings
            activations = [
                hidden_states[i + 1] for i in self.clip_seg_config.extract_layers
            ]

        # step 2: compute conditional embeddings, either from text
        conditional_embeddings = self.get_conditional_embeddings(
            batch_size=pixel_values.shape[0],
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # step 3: forward both the pooled output and the activations through the lightweight decoder to predict masks
        clipseg_decoder_outputs = self.decoder(activations, conditional_embeddings)
        
#------------------------------------------------------------------------------
#-------DATA ADAPTER ----------------------------------------------------------

        a = activations[0]
        ar = self.reduce(a)
        csa=conditional_embeddings.unsqueeze(1)
        a = self.film_mul(csa) * ar + self.film_add(csa)

        ar = a.permute(0,2,1)
        size = int(math.sqrt(ar.shape[2]))

        a = self.reduce2(ar).view(pixel_values.shape[0], self.reduce2(ar).shape[1], size, size)

        at = self.trans_conv1(a)

        at = nnf.interpolate(at, (14,14), mode='bilinear', align_corners=True)

        at = nnf.interpolate(at, pixel_values.shape[2:], mode='bilinear', align_corners=True)
        clipseg_out=self.trans_conv1(a)
       
        decoder_outputs_clipseg = clipseg_decoder_outputs
        decoder_outputs = self.decoder2(pixel_values, a, clipseg_out, a)
        logits = decoder_outputs        
        
        ot=decoder_outputs_clipseg.logits.unsqueeze(1)
        print('ot shape =',ot.shape )
        ot=nnf.interpolate(ot, (224,224), mode='bilinear', align_corners=True)
        biomedclipseg_out=self.convlast(torch.cat([ot,logits],dim=1))
        
        a, clipseg_out=self.clip_seg(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
        
        out=self.convlast(torch.cat([biomedclipseg_out, clipseg_out],dim=1))
        
        return out


class my_CLIPSeg(nn.Module):
    r"""CLIPSeg Official implementation from HuggingFace.

    Args:
        clipseg_hf_api (str): HuggingFace api to import the CLIPSeg implementation; Eg:'CIDAS/clipseg-rd64-refined'
        freeze_encoder (bool): Whether or not to freeze the encoders of pretrained CLIPSeg; Default is False.
        freeze_decoder (bool): Whether or not to freeze the decoder of pretrained CLIPSeg; Default is False.
    """
    def __init__(
        self,
        clipseg_hf_api: str = 'CIDAS/clipseg-rd64-refined',
        freeze_encoder: bool = False,
        freeze_decoder: bool = False,
        maxtext=256,
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
        self.reduce2 = nn.Linear(197, 196)
        
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
        r"""
        Args:
            pixel_values: Normalized image tensor.
            input_ids: Tokenized text input.
            attention_mask: Mask for token inputs, used in the attention layers.

        Returns: Tensor with segmentation logits
        """
        
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
        
        decoder_outputs = self.decoder2(pixel_values, a, decoder_output[0].unsqueeze(1), a)

        return a, self.convlast(torch.cat([decoder_output[0].unsqueeze(1),decoder_outputs],dim=1))


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
        
        self.upconv41 = nn.Conv2d(768,1240,1)
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
        self.eca3 = ECALayer(802)
        
        self.Dropout = nn.Dropout(0.1)

    def forward(self, x, low_level_feat, at, a_my_CLIPseg):
        
        x1 = self.Dropout(self.relu(self.bn1(self.conv12(self.conv11(x)))))
        x1 = self.maxpool(x1)

        x2 = self.Dropout(self.relu(self.bn2(self.conv222(self.conv22(x1)))))
        x2 = self.maxpool(x2)

        x3 = self.Dropout(self.relu(self.bn3(self.conv333(self.conv33(x2)))))
        x3 = self.maxpool(x3)

        x4 = self.Dropout(self.relu(self.bn4(self.conv444(self.conv44(x3)))))
        x4 = self.maxpool(x4)

        xd4=self.relu(self.upbn4(self.convup4(self.convup4(self.upconv4((self.eca4(self.upconv41( torch.cat([low_level_feat, x4, a_my_CLIPseg],dim=1) ))))))))

        xd4 = torch.cat([xd4, x3], dim=1)
        
        xd3=self.relu(self.upbn3(self.convup3(self.convup3((self.upconv3(xd4))))))
        xd3 = torch.cat([xd3, x2], dim=1)
        
        xd2=self.relu(self.upbn2(self.convup2(self.convup2(self.upconv2( xd3 )))))
        xd2 = torch.cat([xd2, x1], dim=1)
        
        xd1=self.relu(self.upbn1(self.convup1(self.convup1(self.upconv1( xd2  )))))       
        
        return self.outconv2(torch.cat([at, self.outconv(xd1)],dim=1))
    
