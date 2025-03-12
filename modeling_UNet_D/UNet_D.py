# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 15:53:07 2025

@author: Julia Dietlmeier <julia.dietlmeier@insight-centre.org>
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import math


resize_transform1 = transforms.Resize((224, 224))
resize_transform2 = transforms.Resize((112, 112))

    
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

class UNet_D(nn.Module):
    def __init__(self, num_classes, BatchNorm):

        super().__init__()
        
        self.num_classes = num_classes
        
        self.relu = nn.ReLU(inplace=True)

        self.conv11 = nn.Conv2d(3, 64, 3, padding=1, bias=True)
        self.conv12 = nn.Conv2d(64, 64, 3, padding=1, bias=True)

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

        
        self.upconv41 = nn.Conv2d(512,1240,1)
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
        
        self.outconv = nn.Conv2d(45, self.num_classes, kernel_size=1)
        self.outconv2 = nn.Conv2d(2, self.num_classes, kernel_size=1)
        
        self.eca4 = ECALayer(1240)
               
        self.Dropout = nn.Dropout(0.1)


    def forward(self, x):

        x0=self.conv11(x)
        x1 = self.Dropout(self.relu(self.bn1(self.conv12(x0))))
        x1 = self.maxpool(x1)

        x2 = self.Dropout(self.relu(self.bn2(self.conv222(self.conv22(x1)))))
        x2 = self.maxpool(x2)

        x3 = self.Dropout(self.relu(self.bn3(self.conv333(self.conv33(x2)))))
        x3 = self.maxpool(x3)

        x4 = self.Dropout(self.relu(self.bn4(self.conv444(self.conv44(x3)))))
        x4 = self.maxpool(x4)

        xd4=self.relu(self.upbn4(self.convup4(self.convup4(self.upconv4((self.eca4(self.upconv41( x4 ) )))))))
        
        xd4 = torch.cat([xd4, x3], dim=1)
        
        xd3=self.relu(self.upbn3(self.convup3(self.convup3((self.upconv3(xd4))))))
        xd3 = torch.cat([xd3, x2], dim=1)
        
        xd2=self.relu(self.upbn2(self.convup2(self.convup2(self.upconv2( xd3 )))))
        xd2 = torch.cat([xd2, x1], dim=1)
        
        xd1=self.relu(self.upbn1(self.convup1(self.convup1(self.upconv1( xd2  )))))

        return self.outconv(xd1)
 
