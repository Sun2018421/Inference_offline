from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from submodule import *
import math
import torch_mlu
import torch_mlu.core.mlu_model as ct


class sparse_convolution(nn.Module):
    '''
        implement the sparse convolution defined in the paper "Sparse Invariant CNNs"
    '''
    def __init__(self):
        super(sparse_convolution, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=11, padding=5, stride=1, bias=False), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=7, padding=3, stride=1, bias=False), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=5, padding=2, stride=1, bias=False), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1, bias=False), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1, bias=False), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(16, 1, kernel_size=1, padding=0, stride=1, bias=False), nn.ReLU(inplace=True))

        self.maxpool1 = nn.MaxPool2d(11, stride = 1, padding = 5)
        self.maxpool2 = nn.MaxPool2d(7, stride = 1, padding = 3)
        self.maxpool3 = nn.MaxPool2d(5, stride = 1, padding = 2)
        self.maxpool4 = nn.MaxPool2d(3, stride = 1, padding = 1)
        self.maxpool5 = nn.MaxPool2d(3, stride = 1, padding = 1)
        self.maxpool6 = nn.MaxPool2d(1, stride = 1)

        self.conv_layer_mask_f = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=11, padding=5, stride=1, bias = False)
        # self.conv_layer_mask11 = nn.Conv2d(in_channels=mask11.shape[1], out_channels=1, kernel_size=7, padding=3, stride=1, bias = False)
        self.conv_layer_mask11 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, padding=3, stride=1, bias = False)
        self.conv_layer_mask7 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, padding=2, stride=1, bias = False)
        self.conv_layer_mask5 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=1, bias = False)
        self.conv_layer_mask3_1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=1, bias = False)
        self.conv_layer_mask3_2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, padding=0, stride=1, bias = False)

        self.small_value = 0.001

    def forward(self, sparse, mask):
        mask_f = mask.to(torch.float)
        sparse_mask = sparse * mask_f
        feature11 = self.conv1(sparse_mask)
        mask11 = self.maxpool1(mask_f)


        # cpu上使用
        # mask11_norm = 1 / (F.conv2d(mask_f, torch.ones(1,mask_f.size()[1],11, 11).to(ct.mlu_device()), padding=5)  + self.small_value)

        # conv_layer = nn.Conv2d(in_channels=mask_f.shape[1], out_channels=1, kernel_size=11, padding=5, stride=1, bias = False)
        # with torch.no_grad():
        #     conv_layer.weight.data.fill_(1.0).to(ct.mlu_device())
        mask11_norm = 1 / (self.conv_layer_mask_f(mask_f) +  self.small_value)

        # mask11_norm = 1 / (F.conv2d(mask_f, torch.ones(1,1,11, 11), padding=5)  + self.small_value)
        #print(mask.size())
        #print(mask11.size())
        #print(feature11.size())
        #print(mask11_norm.size())
        feature11_norm = feature11 * mask11_norm
        #print(feature11_norm.size())
        #print(mask11.size())
        sparse_mask7 = feature11_norm * mask11
        #print(sparse_mask7.size())
        feature7 = self.conv2(sparse_mask7)
        mask7 = self.maxpool2(mask11)
        #cpu上使用
        # mask7_norm = 1 / (F.conv2d(mask11, torch.ones(1,mask11.size()[1],7, 7, device = device), padding=3) + self.small_value)
        # mask7_norm = 1 / (F.conv2d(mask11, torch.ones(1,mask11.size()[1],7, 7), padding=3) + self.small_value)

        # conv_layer = nn.Conv2d(in_channels=mask11.shape[1], out_channels=1, kernel_size=7, padding=3, stride=1, bias = False)
        # with torch.no_grad():
        #     conv_layer.weight.data.fill_(1.0)
        mask7_norm = 1 / (self.conv_layer_mask11(mask11) + self.small_value)

        feature7_norm = feature7 * mask7_norm

        sparse_mask5 = feature7_norm * mask7
        feature5 = self.conv3(sparse_mask5)
        mask5 = self.maxpool3(mask7)

        # cpu上使用
        # mask5_norm = 1 / (F.conv2d(mask7, torch.ones(1,mask7.size()[1],5, 5,device = device), padding=2)  + self.small_value)
        # mask5_norm = 1 / (F.conv2d(mask7, torch.ones(1,mask7.size()[1],5, 5), padding=2)  + self.small_value)
        # conv_layer = nn.Conv2d(in_channels=mask7.shape[1], out_channels=1, kernel_size=5, padding=2, stride=1, bias = False)
        # with torch.no_grad():
        #     conv_layer.weight.data.fill_(1.0)
        mask5_norm = 1 / (self.conv_layer_mask7(mask7) + self.small_value)

        feature5_norm = feature5 * mask5_norm

        sparse_mask3_1 = feature5_norm * mask5
        feature3_1 = self.conv4(sparse_mask3_1)
        mask3_1 = self.maxpool4(mask5)
         # cpu上使用
        # mask3_1_norm = 1 / (F.conv2d(mask5, torch.ones(1,mask5.size()[1],3, 3,device = device), padding=1) + self.small_value)
        # conv_layer = nn.Conv2d(in_channels=mask5.shape[1], out_channels=1, kernel_size=3, padding=1, stride=1, bias = False)
        # with torch.no_grad():
        #     conv_layer.weight.data.fill_(1.0)
        mask3_1_norm = 1 / (self.conv_layer_mask5(mask5) + self.small_value)
        # mask3_1_norm = 1 / (F.conv2d(mask5, torch.ones(1,mask5.size()[1],3, 3), padding=1) + self.small_value)
        feature3_1_norm = feature3_1 * mask3_1_norm

        sparse_mask3_2 = feature3_1_norm * mask3_1
        feature3_2 = self.conv5(sparse_mask3_2)
        mask3_2 = self.maxpool5(mask3_1)

         # cpu上使用
        # mask3_2_norm = 1 / (F.conv2d(mask3_1, torch.ones(1,mask3_1.size()[1],3, 3,device = device), padding=1) + self.small_value)
        # conv_layer = nn.Conv2d(in_channels=mask3_1.shape[1], out_channels=1, kernel_size=3, padding=1, stride=1, bias = False)
        # with torch.no_grad():
        #     conv_layer.weight.data.fill_(1.0)
        mask3_2_norm = 1 / (self.conv_layer_mask3_1(mask3_1) + self.small_value)
        # mask3_2_norm = 1 / (F.conv2d(mask3_1, torch.ones(1,mask3_1.size()[1],3, 3), padding=1) + self.small_value)

        feature3_2_norm = feature3_2 * mask3_2_norm

        sparse_mask1 = feature3_2_norm * mask3_2
        feature1 = self.conv6(sparse_mask1)

         # cpu上使用
        # mask1_norm = 1 / (F.conv2d(mask3_2, torch.ones(1,mask3_2.size()[1],1, 1,device = device), padding=0) + self.small_value)
        # conv_layer = nn.Conv2d(in_channels=mask3_2.shape[1], out_channels=1, kernel_size=1, padding=0, stride=1, bias = False)
        # with torch.no_grad():
        #     conv_layer.weight.data.fill_(1.0)
        mask1_norm = 1 / (self.conv_layer_mask3_2(mask3_2) + self.small_value)
        # mask1_norm = 1 / (F.conv2d(mask3_2, torch.ones(1,mask3_2.size()[1],1, 1), padding=0) + self.small_value)
        feature1_norm = feature1 * mask1_norm

        return feature1_norm

class feature_extraction(nn.Module):
    def __init__(self, concat_feature=False, concat_feature_channel=12):
        super(feature_extraction, self).__init__()
        self.concat_feature = concat_feature

        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),      # convbn(3, 32, 3, 2, 1, 1)
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),   # convbn(32, 32, 3, 1, 1, 1)
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),   # convbn(32, 32, 3, 1, 1, 1)
                                       nn.ReLU(inplace=True))

        # self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 1, 1, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 1, 2, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 192, 1, 2, 1, 1)
        self.layer5 = self._make_layer(BasicBlock, 256, 1, 2, 1, 1)
        self.layer6 = self._make_layer(BasicBlock, 512, 1, 2, 1, 1)
        self.pyramid_pooling = pyramidPooling(512, None, fusion_mode='sum', model_name='icnet')
        self.upconv6 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     convbn(512, 256, 3, 1, 1, 1),      # convbn(512, 256, 3, 1, 1, 1)
                                     nn.ReLU(inplace=True))
        self.iconv5 = nn.Sequential(convbn(512, 256, 3, 1, 1, 1),
                                    nn.ReLU(inplace=True))
        self.upconv5 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     convbn(256, 192, 3, 1, 1, 1),
                                     nn.ReLU(inplace=True))
        self.iconv4 = nn.Sequential(convbn(384, 192, 3, 1, 1, 1),
                                    nn.ReLU(inplace=True))
        self.upconv4 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     convbn(192, 128, 3, 1, 1, 1),
                                     nn.ReLU(inplace=True))
        self.iconv3 = nn.Sequential(convbn(256, 128, 3, 1, 1, 1),
                                    nn.ReLU(inplace=True))
        self.upconv3 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     convbn(128, 64, 3, 1, 1, 1),
                                     nn.ReLU(inplace=True))
        self.iconv2 = nn.Sequential(convbn(128, 64, 3, 1, 1, 1),
                                    nn.ReLU(inplace=True))
        # self.upconv2 = nn.Sequential(nn.Upsample(scale_factor=2),
        #                              convbn(64, 32, 3, 1, 1, 1),
        #                              nn.ReLU(inplace=True))

        # self.gw1 = nn.Sequential(convbn(32, 40, 3, 1, 1, 1),
        #                          nn.ReLU(inplace=True),
        #                          nn.Conv2d(40, 40, kernel_size=1, padding=0, stride=1,
        #                                    bias=False))

        self.gw2 = nn.Sequential(convbn(64, 80, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(80, 80, kernel_size=1, padding=0, stride=1,
                                                    bias=False))

        self.gw3 = nn.Sequential(convbn(128, 160, 3, 1, 1, 1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(160, 160, kernel_size=1, padding=0, stride=1,
                                           bias=False))

        self.gw4 = nn.Sequential(convbn(192, 160, 3, 1, 1, 1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(160, 160, kernel_size=1, padding=0, stride=1,
                                           bias=False))

        self.gw5 = nn.Sequential(convbn(256, 320, 3, 1, 1, 1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(320, 320, kernel_size=1, padding=0, stride=1,
                                           bias=False))

        self.gw6 = nn.Sequential(convbn(512, 320, 3, 1, 1, 1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(320, 320, kernel_size=1, padding=0, stride=1,
                                           bias=False))

        if self.concat_feature:
            # self.concat1 = nn.Sequential(convbn(32, 16, 3, 1, 1, 1),
            #                              nn.ReLU(inplace=True),
            #                              nn.Conv2d(16, concat_feature_channel // 4, kernel_size=1, padding=0, stride=1,
            #                                        bias=False))

            self.concat2 = nn.Sequential(convbn(64, 32, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(32, concat_feature_channel // 2, kernel_size=1, padding=0, stride=1,
                                                    bias=False))
            self.concat3 = nn.Sequential(convbn(128, 128, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                    bias=False))

            self.concat4 = nn.Sequential(convbn(192, 128, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                    bias=False))

            self.concat5 = nn.Sequential(convbn(256, 128, 3, 1, 1, 1),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                   bias=False))

            self.concat6 = nn.Sequential(convbn(512, 128, 3, 1, 1, 1),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                   bias=False))





    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        # x = self.layer1(x)
        l2 = self.layer2(x)     #1/2
        l3 = self.layer3(l2)    #1/4
        l4 = self.layer4(l3)    #1/8
        l5 = self.layer5(l4)    #1/16
        l6 = self.layer6(l5)    #1/32
        l6 = self.pyramid_pooling(l6)

        concat5 = torch.cat((l5, self.upconv6(l6)), dim=1)
        decov_5 = self.iconv5(concat5)
        concat4 = torch.cat((l4, self.upconv5(decov_5)), dim=1)
        # concat4 = torch.cat((l4, self.upconv5(l5)), dim=1)
        decov_4 = self.iconv4(concat4)
        concat3 = torch.cat((l3, self.upconv4(decov_4)), dim=1)
        decov_3 = self.iconv3(concat3)
        concat2 = torch.cat((l2, self.upconv3(decov_3)), dim=1)
        decov_2 = self.iconv2(concat2)
        # decov_1 = self.upconv2(decov_2)


        # gw1 = self.gw1(decov_1)
        gw2 = self.gw2(decov_2)
        gw3 = self.gw3(decov_3)
        gw4 = self.gw4(decov_4)
        gw5 = self.gw5(decov_5)
        gw6 = self.gw6(l6)

        if not self.concat_feature:
            return {"gw2": gw2, "gw3": gw3, "gw4": gw4}
        else:
            # concat_feature1 = self.concat1(decov_1)
            concat_feature2 = self.concat2(decov_2)
            concat_feature3 = self.concat3(decov_3)
            concat_feature4 = self.concat4(decov_4)
            concat_feature5 = self.concat5(decov_5)
            concat_feature6 = self.concat6(l6)
            return {"gw2": gw2, "gw3": gw3, "gw4": gw4, "gw5": gw5, "gw6": gw6,
                    "concat_feature2": concat_feature2, "concat_feature3": concat_feature3, "concat_feature4": concat_feature4,
                    "concat_feature5": concat_feature5, "concat_feature6": concat_feature6}

class hourglassup(nn.Module):
    def __init__(self, in_channels):
        super(hourglassup, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, in_channels * 2, kernel_size=3, stride=2,
                                   padding=1, bias=False)

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Conv3d(in_channels * 2, in_channels * 4, kernel_size=3, stride=2,
                               padding=1, bias=False)

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv8 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.combine1 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        self.combine2 = nn.Sequential(convbn_3d(in_channels * 6, in_channels * 4, 3, 1, 1),
                                      nn.ReLU(inplace=True))
        self.combine3 = nn.Sequential(convbn_3d(in_channels * 6, in_channels * 4, 3, 1, 1),
                                      nn.ReLU(inplace=True))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)
        self.redir3 = convbn_3d(in_channels * 4, in_channels * 4, kernel_size=1, stride=1, pad=0)


    def forward(self, x, feature4, feature5):
        conv1 = self.conv1(x)          #1/8
        conv1 = torch.cat((conv1, feature4), dim=1)   #1/8
        conv1 = self.combine1(conv1)   #1/8
        conv2 = self.conv2(conv1)      #1/8

        conv3 = self.conv3(conv2)      #1/16
        conv3 = torch.cat((conv3, feature5), dim=1)   #1/16
        conv3 = self.combine2(conv3)   #1/16
        conv4 = self.conv4(conv3)      #1/16

        # conv8 = FMish(self.conv8(conv4) + self.redir2(conv2))
        # conv9 = FMish(self.conv9(conv8) + self.redir1(x))

        conv8 = F.relu(self.conv8(conv4) + self.redir2(conv2), inplace=True)
        conv9 = F.relu(self.conv9(conv8) + self.redir1(x), inplace=True)



        return conv9

class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        # conv5 = FMish(self.conv5(conv4) + self.redir2(conv2))
        # conv6 = FMish(self.conv6(conv5) + self.redir1(x))

        return conv6

class cfnet(nn.Module):
    def __init__(self, maxdisp, use_concat_volume=False):
        super(cfnet, self).__init__()
        self.maxdisp = maxdisp
        self.use_concat_volume = use_concat_volume
        self.v_scale_s1 = 1
        self.v_scale_s2 = 2
        self.v_scale_s3 = 3
        self.sample_count_s1 = 6
        self.sample_count_s2 = 10
        self.sample_count_s3 = 14
        self.num_groups = 40
        self.sparse_feature_on = True
        self.uniform_sampler = UniformSampler()
        self.spatial_transformer = SpatialTransformer()

        self.sparse_convolution = sparse_convolution()
        if self.sparse_feature_on:
            self.sparse_channel = 1
        else:
            self.sparse_channel = 0

        if self.use_concat_volume:
            self.concat_channels = 12
            self.feature_extraction = feature_extraction(concat_feature=True,
                                                         concat_feature_channel=self.concat_channels)
        else:
            self.concat_channels = 0
            self.feature_extraction = feature_extraction(concat_feature=False)

        self.dres0 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels*2+self.sparse_channel, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))


        self.dres0_5 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels*2+self.sparse_channel, 64, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(64, 64, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1_5 = nn.Sequential(convbn_3d(64, 64, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(64, 64, 3, 1, 1))

        self.dres0_6 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels*2+self.sparse_channel, 64, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(64, 64, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1_6 = nn.Sequential(convbn_3d(64, 64, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(64, 64, 3, 1, 1))

        self.combine1 = hourglassup(32)

        # self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        # self.dres4 = hourglass(32)

        self.confidence0_s3 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels*2 + 1+self.sparse_channel , 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.confidence1_s3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.confidence2_s3 = hourglass(32)

        self.confidence3_s3 = hourglass(32)

        self.confidence0_s2 = nn.Sequential(convbn_3d(self.num_groups//2 + self.concat_channels + 1+self.sparse_channel, 16, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(16, 16, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.confidence1_s2 = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(16, 16, 3, 1, 1))

        self.confidence2_s2 = hourglass(16)

        self.confidence3_s2 = hourglass(16)


        # self.confidence0_s1 = nn.Sequential(convbn_3d(self.num_groups // 4 + self.concat_channels // 2 + 1, 16, 3, 1, 1),
        #                                     nn.ReLU(inplace=True),
        #                                     convbn_3d(16, 16, 3, 1, 1),
        #                                     nn.ReLU(inplace=True))
        #
        # self.confidence1_s1 = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
        #                                     nn.ReLU(inplace=True),
        #                                     convbn_3d(16, 16, 3, 1, 1))
        #
        # self.confidence2_s1 = hourglass(16)

        # self.confidence3 = hourglass(32)
        #
        # self.confidence4 = hourglass(32)

        self.confidence_classif0_s3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                                    nn.ReLU(inplace=True),
                                                    nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.confidence_classif1_s3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.confidence_classifmid_s3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                                    nn.ReLU(inplace=True),
                                                    nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.confidence_classif0_s2 = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
                                                    nn.ReLU(inplace=True),
                                                    nn.Conv3d(16, 1, kernel_size=3, padding=1, stride=1, bias=False))


        self.confidence_classif1_s2 = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
                                                    nn.ReLU(inplace=True),
                                                    nn.Conv3d(16, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.confidence_classifmid_s2 = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
                                                    nn.ReLU(inplace=True),
                                                    nn.Conv3d(16, 1, kernel_size=3, padding=1, stride=1, bias=False))

        # self.confidence_classif1_s1 = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
        #                                             nn.ReLU(inplace=True),
        #                                             nn.Conv3d(16, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.gamma_s3 = nn.Parameter(torch.zeros(1))
        self.beta_s3 = nn.Parameter(torch.zeros(1))
        self.gamma_s2 = nn.Parameter(torch.zeros(1))
        self.beta_s2 = nn.Parameter(torch.zeros(1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
                # m.bias.data.zero_()

    def generate_search_range(self, sample_count, input_min_disparity, input_max_disparity, scale):
        """
        Description:    Generates the disparity search range.

        Returns:
            :min_disparity: Lower bound of disparity search range
            :max_disparity: Upper bound of disaprity search range.
        """
        # have been modified, origin: max=self.maxdisp
        assert input_max_disparity.shape == input_min_disparity.shape
        # sample_count = sample_count * torch.ones(input_max_disparity.shape).to(ct.mlu_device())

        min_disparity = torch.clamp(input_min_disparity - torch.clamp((
                - (input_max_disparity - sample_count) + input_min_disparity), min=0) / 2.0, min=0, max=self.maxdisp // (2**scale) - 1)
        max_disparity = torch.clamp(input_max_disparity + torch.clamp(
                - (input_max_disparity - sample_count) + input_min_disparity, min=0) / 2.0, min=0, max=self.maxdisp // (2**scale) - 1)

        return min_disparity, max_disparity

    def generate_disparity_samples(self, min_disparity, max_disparity, sample_count=12):
        """
        Description:    Generates "sample_count" number of disparity samples from the
                                                            search range (min_disparity, max_disparity)
                        Samples are generated by uniform sampler

        Args:
            :min_disparity: LowerBound of the disaprity search range.
            :max_disparity: UpperBound of the disparity search range.
            :sample_count: Number of samples to be generated from the input search range.

        Returns:
            :disparity_samples:
        """
        disparity_samples = self.uniform_sampler(min_disparity, max_disparity, sample_count)

        # disparity_samples = torch.cat((torch.floor(min_disparity), disparity_samples, torch.ceil(max_disparity)),
        #                               dim=1).long()                   # disparity level = sample_count + 2
        disparity_samples = torch.cat((torch.floor(min_disparity), disparity_samples, torch.round(max_disparity + 0.5)), dim=1)
        return disparity_samples

    def cost_volume_generator(self, left_input, right_input, disparity_samples, model = 'concat', num_groups = 40):
        """
        Description: Generates cost-volume using left image features, disaprity samples
                                                            and warped right image features.
        Args:
            :left_input: Left Image fetaures
            :right_input: Right Image features
            :disparity_samples: Disaprity samples
            :model : concat or group correlation

        Returns:
            :cost_volume:
            :disaprity_samples:
        """

        right_feature_map, left_feature_map = self.spatial_transformer(left_input, right_input, disparity_samples)
        disparity_samples = disparity_samples.unsqueeze(1)  # .float()
        if model == 'concat':
             cost_volume = torch.cat((left_feature_map, right_feature_map), dim=1)
        else:
             cost_volume = groupwise_correlation_4D(left_feature_map, right_feature_map, num_groups)
        return cost_volume, disparity_samples

    def sparse_downsample(self, sparse, sparse_mask):
        '''
           try the downsample by sample the sparse result by fixed step
        '''

        sparse_out = {}
        # print(sparse.cpu().shape, sparse_mask.cpu().shape)

        # [1, 1, 512, 1024]

        sparse_out["sparse2"] = torch.div(sparse.reshape(-1,2)[:,0].reshape(256,-1)[:,:512].reshape(1, 1, 256,512), 2)
        sparse_out["sparse3"] = torch.div(sparse.reshape(-1,4)[:,0].reshape(128,-1)[:,:256].reshape(1, 1, 128,256), 4)
        sparse_out["sparse4"] = torch.div(sparse.reshape(-1,8)[:,0].reshape(64,-1)[:,:128].reshape(1, 1, 64,128), 8)
        sparse_out["sparse5"] = torch.div(sparse.reshape(-1,16)[:,0].reshape(32,-1)[:,:64].reshape(1, 1, 32,64), 16)
        sparse_out["sparse6"] = torch.div(sparse.reshape(-1,32)[:,0].reshape(16,-1)[:,:32].reshape(1, 1, 16,32), 32)
        sparse_mask_out = {}
        sparse_mask_out["sparse_mask2"] = sparse_mask.reshape(-1,2)[:,0].reshape(256,-1)[:,:512].reshape(1, 1, 256,512)
        sparse_mask_out["sparse_mask3"] = sparse_mask.reshape(-1,4)[:,0].reshape(128,-1)[:,:256].reshape(1, 1, 128,256)
        sparse_mask_out["sparse_mask4"] = sparse_mask.reshape(-1,8)[:,0].reshape(64,-1)[:,:128].reshape(1, 1, 64,128)
        sparse_mask_out["sparse_mask5"] = sparse_mask.reshape(-1,16)[:,0].reshape(32,-1)[:,:64].reshape(1, 1, 32,64)
        sparse_mask_out["sparse_mask6"] = sparse_mask.reshape(-1,32)[:,0].reshape(16,-1)[:,:32].reshape(1, 1, 16,32)
        return sparse_out, sparse_mask_out

    def sparse_expand(self, sparse, sparse_mask, left_img):
        '''
            RGB guieded sparse expand
            input:
                sparse_mask: N*1*W*H  [1, 1, 512, 1024]
                sparse:      N*1*W*H  [1, 1, 512, 1024]
                left_img:    N*3*W*H
            output:
                sparse_mask_exp: N*1*W*H
                sparse_exp:      N*1*W*H
        '''
        print("===================in_sparse_expand==============================")
        threshold = 10.0
        # 以下均为torch.Size([1, 3, 510, 1022])
        img_center = left_img[:,:,1:-1,1:-1]
        img_up_left = left_img[:,:,0:-2,0:-2]
        img_up = left_img[:,:,1:-1,0:-2]
        img_up_right = left_img[:,:,2:,0:-2]
        img_left = left_img[:,:,0:-2,1:-1]
        img_right = left_img[:,:,2:,1:-1]
        img_down_left = left_img[:,:,0:-2,2:]
        img_down = left_img[:,:,1:-1,2:]
        img_down_right = left_img[:,:,2:,2:]
        # mask_ones = torch.ones([1, 1, 510, 1022]).half().to(ct.mlu_device())
        # mask_ones = torch.ones([1, 1, 510, 1022]).to(ct.mlu_device())


        # print(torch.sum(torch.abs(img_up_left - img_center),1).cpu().shape, torch.sum(torch.abs(img_up_left - img_center),1).cpu())  # torch.Size([1, 510, 1022])
        mask_up_left = torch.unsqueeze(torch.threshold(-torch.threshold(torch.sum(torch.abs(img_up_left - img_center), 1), threshold, -1.0), 0.0, 0.0), dim=1) \
                                                    * torch.threshold(-torch.threshold(sparse_mask[:,:,0:-2,0:-2], 1.0, -1.0), 0.0, 0.0) \
                                                    * sparse_mask[:,:,1:-1,1:-1]  # torch.Size([1, 1, 510, 1022])
        new_mask_up_left = - (mask_up_left - 1)
        sparse = torch.cat([torch.cat([sparse[:,:,0:-2,0:-2] *new_mask_up_left  + sparse[:,:,1:-1,1:-1] * mask_up_left, sparse[:,:,0:-2,1022:1024]], dim=3), sparse[:,:,510:,:]], dim=2)
        # sparse[:,:,0:510,0:1022][mask_up_left] = sparse[:,:,1:511,1:1023][mask_up_left]
        sparse_mask = torch.cat([torch.cat([sparse_mask[:,:,0:-2,0:-2] * new_mask_up_left + sparse_mask[:,:,1:-1,1:-1] * mask_up_left, sparse_mask[:,:,0:-2,1022:1024] * 1.0], dim=3), sparse_mask[:,:,510:,:] * 1.0], dim=2)
        # sparse_mask[:,:,0:-2,0:-2][mask_up_left] = sparse_mask[:,:,1:-1,1:-1][mask_up_left]

        mask_up = torch.unsqueeze(torch.threshold(-torch.threshold(torch.sum(torch.abs(img_up - img_center), 1), threshold, -1.0), 0.0, 0.0), dim=1) \
                                                    * torch.threshold(-torch.threshold(sparse_mask[:,:,1:-1,0:-2], 1.0, -1.0), 0.0, 0.0) \
                                                    * sparse_mask[:,:,1:-1,1:-1] # [1, 1, 510, 1022]
        new_mask_up = - (mask_up - 1)
        sparse = torch.cat([sparse[:,:,0:1,:], torch.cat([torch.cat([sparse[:,:,1:-1,0:-2] *new_mask_up  + sparse[:,:,1:-1,1:-1] * mask_up, sparse[:,:,1:-1,1022:]], dim=3), sparse[:,:,511:,:]], dim=2)], dim=2)
        # sparse[:,:,1:-1,0:-2][mask_up] = sparse[:,:,1:-1,1:-1][mask_up]
        sparse_mask = torch.cat([sparse_mask[:,:,0:1,:], torch.cat([torch.cat([sparse_mask[:,:,1:-1,0:-2] *new_mask_up  + sparse_mask[:,:,1:-1,1:-1] * mask_up, sparse_mask[:,:,1:-1,1022:]], dim=3), sparse_mask[:,:,511:,:]], dim=2)], dim=2)
        # sparse_mask[:,:,1:-1,0:-2][mask_up] = sparse_mask[:,:,1:-1,1:-1][mask_up]

        mask_up_right = torch.unsqueeze(torch.threshold(-torch.threshold(torch.sum(torch.abs(img_up_right - img_center), 1), threshold, -1.0), 0.0, 0.0), dim=1) \
                                                    * torch.threshold(-torch.threshold(sparse_mask[:,:,2:,0:-2], 1.0, -1.0), 0.0, 0.0) \
                                                    * sparse_mask[:,:,1:-1,1:-1]  # [1, 1, 510, 1022]
        new_mask_up_right = - (mask_up_right - 1)
        sparse = torch.cat([sparse[:,:,0:2,:],torch.cat([sparse[:,:,2:,0:-2] * new_mask_up_right + sparse[:,:,1:-1,1:-1] * mask_up_right, sparse[:,:,2:,1022:]], dim=3)], dim=2)
        # sparse[:,:,2:,0:-2][mask_up_right] = sparse[:,:,1:-1,1:-1][mask_up_right]
        sparse_mask = torch.cat([sparse_mask[:,:,0:2,:],torch.cat([sparse_mask[:,:,2:,0:-2] * new_mask_up_right + sparse_mask[:,:,1:-1,1:-1] * mask_up_right, sparse_mask[:,:,2:,1022:]], dim=3)], dim=2)
        # sparse_mask[:,:,2:,0:-2][mask_up_right] = sparse_mask[:,:,1:-1,1:-1][mask_up_right]

        mask_left = torch.unsqueeze(torch.threshold(-torch.threshold(torch.sum(torch.abs(img_left - img_center), 1), threshold, -1.0), 0.0, 0.0), dim=1) \
                                                    * torch.threshold(-torch.threshold(sparse_mask[:,:,0:-2,1:-1], 1.0, -1.0), 0.0, 0.0) \
                                                    * -sparse_mask[:,:,1:-1,1:-1]
        new_mask_left = - (mask_left - 1)
        sparse = torch.cat([torch.cat([torch.cat([sparse[:,:,0:-2,0:1], sparse[:,:,0:-2,1:-1] * new_mask_left + sparse[:,:,1:-1,1:-1] * mask_left],dim=3), sparse[:,:,0:-2,1023:]],dim=3), sparse[:,:,510:,:]],dim=2)
        # sparse[:,:,0:-2,1:-1][mask_left] = sparse[:,:,1:-1,1:-1][mask_left]
        sparse_mask = torch.cat([torch.cat([torch.cat([sparse_mask[:,:,0:-2,0:1], sparse_mask[:,:,0:-2,1:-1] * new_mask_left + sparse_mask[:,:,1:-1,1:-1] * mask_left],dim=3), sparse_mask[:,:,0:-2,1023:]],dim=3), sparse_mask[:,:,510:,:]],dim=2)
        # sparse_mask[:,:,0:-2,1:-1][mask_left] = sparse_mask[:,:,1:-1,1:-1][mask_left]

        mask_right = torch.unsqueeze(torch.threshold(-torch.threshold(torch.sum(torch.abs(img_right - img_center), 1), threshold, -1.0), 0.0, 0.0), dim=1) \
                                                    * torch.threshold(-torch.threshold(sparse_mask[:,:,2:,1:-1] ,1.0, -1.0), 0.0, 0.0) \
                                                    * -sparse_mask[:,:,1:-1,1:-1]
        new_mask_right = - (mask_right - 1)
        sparse = torch.cat([sparse[:,:,0:2,:], torch.cat([sparse[:,:,2:,0:1],torch.cat([sparse[:,:,2:,1:-1] * new_mask_right + sparse[:,:,1:-1,1:-1] * mask_right, sparse[:,:,2:,1023:]],dim=3)],dim=3)], dim=2)
        # sparse[:,:,2:,1:-1][mask_right] = sparse[:,:,1:-1,1:-1][mask_right]
        sparse_mask = torch.cat([sparse[:,:,0:2,:], torch.cat([sparse_mask[:,:,2:,0:1],torch.cat([sparse_mask[:,:,2:,1:-1] * new_mask_right + sparse_mask[:,:,1:-1,1:-1] * mask_right, sparse_mask[:,:,2:,1023:]],dim=3)],dim=3)], dim=2)
        # sparse_mask[:,:,2:,1:-1][mask_right] = sparse_mask[:,:,1:-1,1:-1][mask_right]


        mask_down_left = torch.unsqueeze(torch.threshold(-torch.threshold(torch.sum(torch.abs(img_down_left - img_center), 1), threshold, -1.0), 0.0, 0.0), dim=1) \
                                                   * torch.threshold(-torch.threshold(sparse_mask[:,:,0:-2,2:], 1.0, -1.0), 0.0, 0.0) \
                                                   * sparse_mask[:,:,1:-1,1:-1]
        new_mask_down_left = - (mask_down_left - 1)
        sparse = torch.cat([torch.cat([sparse[:,:,0:-2,0:2],sparse[:,:,0:-2,2:] * new_mask_down_left + sparse[:,:,1:-1,1:-1] * mask_down_left],dim=3), sparse[:,:,510:,:]], dim=2)
        # sparse[:,:,0:-2,2:][mask_down_left] = sparse[:,:,1:-1,1:-1][mask_down_left]
        sparse_mask = torch.cat([torch.cat([sparse_mask[:,:,0:-2,0:2],sparse_mask[:,:,0:-2,2:] * new_mask_down_left + sparse_mask[:,:,1:-1,1:-1] * mask_down_left],dim=3), sparse_mask[:,:,510:,:]], dim=2)
        # sparse_mask[:,:,0:-2,2:][mask_down_left] = sparse_mask[:,:,1:-1,1:-1][mask_down_left]
        # return sparse, sparse_mask

        mask_down = torch.unsqueeze(torch.threshold(-torch.threshold(torch.sum(torch.abs(img_down - img_center), 1), threshold, -1.0), 0.0, 0.0), dim=1) \
                                                   * torch.threshold(-torch.threshold(sparse_mask[:,:,1:-1,2:], 1.0, -1.0), 0.0, 0.0) \
                                                   * sparse_mask[:,:,1:-1,1:-1]
        new_mask_down = - (mask_down - 1)
        sparse = torch.cat([sparse[:,:,0:1,:], torch.cat([torch.cat([sparse[:,:,1:-1,0:2],sparse[:,:,1:-1,2:] * new_mask_down + sparse[:,:,1:-1,1:-1] * mask_down], dim=3), sparse[:,:,511:,:]], dim=2)],dim=2)
        # sparse[:,:,1:-1,2:][mask_down] = sparse[:,:,1:-1,1:-1][mask_down]
        sparse_mask = torch.cat([sparse_mask[:,:,0:1,:], torch.cat([torch.cat([sparse_mask[:,:,1:-1,0:2],sparse_mask[:,:,1:-1,2:] * new_mask_down + sparse_mask[:,:,1:-1,1:-1] * mask_down], dim=3), sparse_mask[:,:,511:,:]], dim=2)],dim=2)
        # sparse_mask[:,:,1:-1,2:][mask_down] = sparse_mask[:,:,1:-1,1:-1][mask_down]
        #return sparse, sparse_mask

        mask_down_right = torch.unsqueeze(torch.threshold(-torch.threshold(torch.sum(torch.abs(img_down_right - img_center), 1), threshold, -1.0), 0.0, 0.0), dim=1) \
                                                  * torch.threshold(-torch.threshold(sparse_mask[:,:,2:,2:], 1.0, -1.0), 0.0, 0.0) \
                                                  * torch.threshold(-torch.threshold(-sparse_mask[:,:,1:-1,1:-1], 0.0, -1.0), 0.0, 0.0)
        new_mask_down_right = - (mask_down_right - 1)
        sparse = torch.cat([sparse[:,:,0:2,:], torch.cat([sparse[:,:,2:,0:2],sparse[:,:,2:,2:] * new_mask_down_right + sparse[:,:,1:-1,1:-1] * mask_down_right], dim=3)], dim=2)
        # sparse[:,:,2:,2:][mask_down_right] = sparse[:,:,1:-1,1:-1][mask_down_right]
        sparse_mask = torch.cat([sparse_mask[:,:,0:2,:], torch.cat([sparse_mask[:,:,2:,0:2],sparse_mask[:,:,2:,2:] * new_mask_down_right + sparse_mask[:,:,1:-1,1:-1] * mask_down_right], dim=3)], dim=2)
        # sparse_mask[:,:,2:,2:][mask_down_right] = sparse_mask[:,:,1:-1,1:-1][mask_down_right]

        return sparse, sparse_mask


    def cost_volum_modulation(self, sparse, sparse_mask, sampler, cost_volum, max_disp = None):
        '''
            input:
                sparse_mask: N*1*W*H
                sparse:      N*1*W*H
                sampler:     N*D*W*H
                cost_volum:  N*2F*D*W*H
            media_data:
                gaussian_modulation_element: N*D*W*H -->N*1*D*W*H
            output:
                modulated_cost_volum: N*2F*D*W*H

            process:
              sparse_mask
                            + sampler --> gaussian_modulation_element * cost_volum --> modulated_cost_volum
              sparse
        '''
        # superparameter k=10,c=1 is set by GSM
        k = 10
        c = 1
        # N*W*H -->N*1*W*H
        #sparse = torch.unsqueeze(sparse, dim=1)
        #sparse_mask = torch.unsqueeze(sparse_mask, dim=1)
        # calculate the modulation element
        #ones_sparse_mask = torch.ones(sparse_mask.shape).half().to(ct.mlu_device())
        #ones_sparse_mask = torch.ones(sparse_mask.shape).to(ct.mlu_device())
        if max_disp is None:
            gaussian_modulation_element = - (sparse_mask - 1) + sparse_mask*k*torch.exp(-torch.pow(sparse - sampler, 2)/(2*pow(c, 2)))
        else:
            sampler_gen = torch.linspace(0, max_disp-1, max_disp).view(1, max_disp, 1, 1).half().to(ct.mlu_device()) * 1.0
            # sampler_gen = torch.linspace(0, max_disp-1, max_disp).view(1, max_disp, 1, 1).to(ct.mlu_device()) * 1.0
            gaussian_modulation_element = - (sparse_mask - 1) + sparse_mask*k*torch.exp(-torch.pow(sparse - sampler_gen, 2)/(2*pow(c, 2)))

        gaussian_modulation_element = torch.unsqueeze(gaussian_modulation_element, dim = 1)
        modulated_cost_volum = gaussian_modulation_element * cost_volum

        return modulated_cost_volum

    def sparse_feature_extraction(self, sparse_out, sparse_mask_out):
        ''' extract sparse feature by sparse convolution '''
        feature_sparse = {}
        feature_sparse['s2'] = torch.unsqueeze(self.sparse_convolution(sparse_out['sparse2'], sparse_mask_out['sparse_mask2']), dim=2)
        feature_sparse['s3'] = torch.unsqueeze(self.sparse_convolution(sparse_out['sparse3'], sparse_mask_out['sparse_mask3']), dim=2)
        feature_sparse['s4'] = torch.unsqueeze(self.sparse_convolution(sparse_out['sparse4'], sparse_mask_out['sparse_mask4']), dim=2)
        feature_sparse['s5'] = torch.unsqueeze(self.sparse_convolution(sparse_out['sparse5'], sparse_mask_out['sparse_mask5']), dim=2)
        feature_sparse['s6'] = torch.unsqueeze(self.sparse_convolution(sparse_out['sparse6'], sparse_mask_out['sparse_mask6']), dim=2)

        return feature_sparse


    def spatial_pre(self, left_y_coordinate, left_input, right_input, disparity_samples, length):
        # input: left_input  right_input disparity_samples
        # output: warped_right_feature_map_left  gather_index  left_feature_map  right_feature_map
        # length = right_input.cpu().size()[3]
        left_feature_map = left_input.unsqueeze(0).repeat(disparity_samples.size()[1], 1, 1, 1, 1).permute([1, 2, 0, 3, 4])  # [1, 12, 16, 128,256]
        right_feature_map = right_input.unsqueeze(0).repeat(disparity_samples.size()[1], 1, 1, 1, 1).permute([1, 2, 0, 3, 4]) # [1, 12, 128, 256]
        # left_y_coordinate = torch.linspace(0.0, left_input.size()[3]-1, left_input.size()[3]).half().to(ct.mlu_device())  # [256]  -> 0.1.2...255
        left_y_coordinate = left_y_coordinate.unsqueeze(0).repeat(left_input.size()[2], 1)  # [128, 256]
        left_y_coordinate = torch.clamp(left_y_coordinate, min=0, max=left_input.size()[3] - 1)  # [128, 256]
        left_y_coordinate = left_y_coordinate.unsqueeze(0).repeat(left_input.size()[0], 1, 1)  # [1, 128, 256]
        right_y_coordinate = left_y_coordinate.unsqueeze(0).repeat(disparity_samples.size()[1], 1, 1, 1).permute([1, 0, 2, 3]) - disparity_samples  # [1, 16, 128, 256] - [1, 16, 128, 256]

        right_y_coordinate_1 = right_y_coordinate
        # right_y_coordinate_1 = right_y_coordinate_1.unsqueeze(1) # [1, 1, 16, 128, 256]
        right_y_coordinate = torch.clamp(right_y_coordinate, min=0, max=right_input.size()[3] - 1)  # [1, 16, 128, 256]
        gather_index = right_y_coordinate.unsqueeze(0).repeat(right_input.size()[1], 1, 1, 1, 1).permute([1, 0, 2, 3, 4]) # [1, 12, 16, 128, 256]

        # return right_y_coordinate_1, gather_index, left_feature_map, right_feature_map

        # TODO
        warped_right_feature_map_left =  - torch.threshold(-torch.threshold(right_y_coordinate_1, 0, -1.0), 0.0, 0.0) 
        warped_right_feature_map_left2 = - torch.threshold(-torch.threshold(right_y_coordinate_1, length - 1, 0), -length + 1, 1)
        return warped_right_feature_map_left + warped_right_feature_map_left2 + 1, gather_index, left_feature_map, right_feature_map
        

    def spatial_post(self, right_feature_map, gather_index, warped_right_feature_map_left):
        # right_input  gather_index  warped_right_feature_map_left
        warped_right_feature_map = torch.gather(right_feature_map, dim=4, index=gather_index)
        warped_right_feature_map = warped_right_feature_map_left * warped_right_feature_map
        return warped_right_feature_map

    def forward(self, *input):
        if(len(input) == 1):
            input= input[0]
        if(len(input) == 5):
            return self.forward1(*input)
        elif(len(input) == 15):
            return self.forward2(*input)
        elif(len(input) == 10):
            return self.forward3(*input)

    def forward1(self, *input):
        left, right, sparse, sparse_mask, left_y_coordinate = input
        print('================start==================')
        features_left = self.feature_extraction(left)
        features_right = self.feature_extraction(right)
        # RGB guided sparse expanded
        sparse, sparse_mask = self.sparse_expand(sparse, sparse_mask, left)
        print("====================sparse_expand=====================")
        sparse_out, sparse_mask_out = self.sparse_downsample(sparse, sparse_mask)
        feature_sparse = self.sparse_feature_extraction(sparse_out, sparse_mask_out)

        gwc_volume4 = build_gwc_volume(features_left["gw4"], features_right["gw4"], self.maxdisp // 8, self.num_groups)
        gwc_volume5 = build_gwc_volume(features_left["gw5"], features_right["gw5"], self.maxdisp // 16, self.num_groups)
        gwc_volume6 = build_gwc_volume(features_left["gw6"], features_right["gw6"], self.maxdisp // 32, self.num_groups)
        if self.use_concat_volume:
            print('use_concat_volume')
            concat_volume4 = build_concat_volume(features_left["concat_feature4"], features_right["concat_feature4"], self.maxdisp // 8)
            concat_volume5 = build_concat_volume(features_left["concat_feature5"], features_right["concat_feature5"], self.maxdisp // 16)
            concat_volume6 = build_concat_volume(features_left["concat_feature6"], features_right["concat_feature6"], self.maxdisp // 32)
            volume4 = torch.cat((gwc_volume4, concat_volume4), 1)
            volume5 = torch.cat((gwc_volume5, concat_volume5), 1)
            volume6 = torch.cat((gwc_volume6, concat_volume6), 1)
        else:
            volume4 = gwc_volume4
            volume5 = gwc_volume5
            volume6 = gwc_volume6

        if self.sparse_feature_on:
            volume4_s = torch.cat((volume4, feature_sparse['s4'].repeat(1,1,volume4.size()[2],1,1)), 1)
            volume5_s = torch.cat((volume5, feature_sparse['s5'].repeat(1,1,volume5.size()[2],1,1)), 1)
            volume6_s = torch.cat((volume6, feature_sparse['s6'].repeat(1,1,volume6.size()[2],1,1)), 1)
        else:
            volume4_s = volume4
            volume5_s = volume5
            volume6_s = volume6

        volume6_m = self.cost_volum_modulation(sparse_out["sparse6"], sparse_mask_out["sparse_mask6"], None, volume6_s, self.maxdisp // 32)
        volume5_m = self.cost_volum_modulation(sparse_out["sparse5"], sparse_mask_out["sparse_mask5"], None, volume5_s, self.maxdisp // 16)
        volume4_m = self.cost_volum_modulation(sparse_out["sparse4"], sparse_mask_out["sparse_mask4"], None, volume4_s, self.maxdisp // 8)

        # mlu修改
        cost0_4 = self.dres0(volume4_m)
        cost0_4 = self.dres1(cost0_4) + cost0_4
        cost0_5 = self.dres0_5(volume5_m)
        cost0_5 = self.dres1_5(cost0_5) + cost0_5
        cost0_6 = self.dres0_6(volume6_m)
        cost0_6 = self.dres1_6(cost0_6) + cost0_6
        out1_4 = self.combine1(cost0_4, cost0_5, cost0_6)
        out2_4 = self.dres3(out1_4)
        cost2_s4 = self.classif2(out2_4)
        cost2_s4 = torch.squeeze(cost2_s4, 1)
        pred2_possibility_s4 = F.softmax(cost2_s4, dim=1)
        pred2_s4 = disparity_regression(pred2_possibility_s4, self.maxdisp // 8).unsqueeze(1)
        pred2_s4_cur = pred2_s4.detach()
        pred2_v_s4 = disparity_variance(pred2_possibility_s4, self.maxdisp // 8, pred2_s4_cur)  # get the variance
        pred2_v_s4 = pred2_v_s4.sqrt()

        #mindisparity_s3 = pred2_s4_cur - (self.gamma_s3 + 1) * pred2_v_s4 - self.beta_s3
        #maxdisparity_s3 = pred2_s4_cur + (self.gamma_s3 + 1) * pred2_v_s4 + self.beta_s3
        mindisparity_s3 = pred2_s4_cur - pred2_v_s4
        maxdisparity_s3 = pred2_s4_cur + pred2_v_s4
        maxdisparity_s3 = F.upsample(maxdisparity_s3 * 2, [left.size()[2] // 4, left.size()[3] // 4], mode='bilinear', align_corners=True)
        mindisparity_s3 = F.upsample(mindisparity_s3 * 2, [left.size()[2] // 4, left.size()[3] // 4], mode='bilinear', align_corners=True)
        mindisparity_s3_1, maxdisparity_s3_1 = self.generate_search_range(self.sample_count_s3 + 1, mindisparity_s3, maxdisparity_s3, scale=2)

        disparity_samples_s3 = self.generate_disparity_samples(mindisparity_s3_1, maxdisparity_s3_1, self.sample_count_s3)  # 本该是long
        warped_right_feature_map_left, gather_index, left_feature_map, right_feature_map = self.spatial_pre(left_y_coordinate, features_left["concat_feature3"], features_right["concat_feature3"], disparity_samples_s3, 256)
        warped_right_feature_map_left2, gather_index2, left_feature_map2, right_feature_map2 = self.spatial_pre(left_y_coordinate, features_left["gw3"], features_right["gw3"], disparity_samples_s3, 256)
        # warped_right_feature_map
        #torch.Size([1, 16, 128, 256])  -->warped_right_feature_map_left
        #torch.Size([1, 12, 16, 128, 256])  --> left_feature_map
        # warped_right_feature_map2
        #torch.Size([1, 16, 128, 256])  --> warped_right_feature_map_left2
        #torch.Size([1, 160, 16, 128, 256])  --> left_feature_map2
        #torch.Size([1, 16, 128, 256])  -->disparity_samples_s3
        #torch.Size([1, 1, 1, 128, 256])
        #torch.Size([1, 1, 128, 256])
        #torch.Size([1, 1, 128, 256])
        #torch.Size([1, 6, 256, 512])
        #torch.Size([1, 6, 256, 512])
        #torch.Size([1, 80, 256, 512])
        #torch.Size([1, 80, 256, 512])
        #torch.Size([1, 1, 1, 256, 512])
        #torch.Size([1, 1, 256, 512])
        #torch.Size([1, 1, 256, 512])
        
        return tuple([warped_right_feature_map_left, gather_index, left_feature_map, right_feature_map, warped_right_feature_map_left2, gather_index2, left_feature_map2, right_feature_map2, \
                disparity_samples_s3, feature_sparse['s3'], sparse_out["sparse3"], sparse_mask_out["sparse_mask3"], \
                features_left["concat_feature2"], features_right["concat_feature2"], features_left["gw2"], features_right["gw2"],  \
                feature_sparse['s2'], sparse_out["sparse2"], sparse_mask_out["sparse_mask2"]])
        
    def forward2(self, *input):
        # =======================================================================================================================================================================
        # input: warped_right_feature_map_left, gather_index, left_feature_map, right_feature_map, warped_right_feature_map_left2, gather_index2, left_feature_map2, right_feature_map2 \
        # disparity_samples_s3, feature_sparse['s3'], sparse_out["sparse3"], sparse_mask_out["sparse_mask3"]
        feature_sparse = {}
        sparse_out = {}
        sparse_mask_out = {}
        features_left = {}
        features_right = {}
        warped_right_feature_map, warped_right_feature_map_left, left_feature_map,  warped_right_feature_map2, warped_right_feature_map_left2,  left_feature_map2,  \
            disparity_samples_s3, feature_sparse['s3'], sparse_out["sparse3"], sparse_mask_out["sparse_mask3"], \
            features_left["concat_feature2"], features_right["concat_feature2"], features_left["gw2"], features_right["gw2"], left_y_coordinate = input

        #warped_right_feature_map = torch.gather(right_feature_map.float(), dim=4, index=gather_index)
        #warped_right_feature_map2 = torch.gather(right_feature_map2.float(), dim=4, index=gather_index2)
        right_feature_map = warped_right_feature_map_left * warped_right_feature_map  # [1，12，16，128，256]
        right_feature_map2 = warped_right_feature_map_left2 * warped_right_feature_map2
        confidence_v_concat_s3 = torch.cat((left_feature_map, right_feature_map), dim=1)
        confidence_v_gwc_s3 = groupwise_correlation_4D(left_feature_map2, right_feature_map2, self.num_groups)

        # confidence_v_concat_s3, _ = self.cost_volume_generator(features_left["concat_feature3"], features_right["concat_feature3"], disparity_samples_s3, 'concat')
        # confidence_v_gwc_s3, disparity_samples_s3 = self.cost_volume_generator(features_left["gw3"], features_right["gw3"], disparity_samples_s3, 'gwc', self.num_groups)
        
        disparity_samples_s3 = disparity_samples_s3.unsqueeze(1)  # .float()
        confidence_v_s3 = torch.cat((confidence_v_gwc_s3, confidence_v_concat_s3, disparity_samples_s3), dim=1)
        # add gaussian modulation
        if self.sparse_feature_on:
            confidence_v_s3_s = torch.cat((confidence_v_s3, feature_sparse['s3'].repeat(1,1,confidence_v_s3.size()[2],1,1)), 1)
        else:
            confidence_v_s3_s = confidence_v_s3
        disparity_samples_s3 = torch.squeeze(disparity_samples_s3, dim=1)
        confidence_v_s3_m = self.cost_volum_modulation(sparse_out["sparse3"], sparse_mask_out["sparse_mask3"], disparity_samples_s3, confidence_v_s3_s)
        cost0_s3 = self.confidence0_s3(confidence_v_s3_m)
        cost0_s3 = self.confidence1_s3(cost0_s3) + cost0_s3
        out1_s3 = self.confidence2_s3(cost0_s3)
        out2_s3 = self.confidence3_s3(out1_s3)
        cost1_s3 = self.confidence_classif1_s3(out2_s3).squeeze(1)
        cost1_s3_possibility = F.softmax(cost1_s3, dim=1)
        pred1_s3 = torch.sum(cost1_s3_possibility * disparity_samples_s3, dim=1, keepdim=True)
        pred1_s3_cur = pred1_s3.detach()
        pred1_v_s3 = disparity_variance_confidence(cost1_s3_possibility, disparity_samples_s3, pred1_s3_cur)
        pred1_v_s3 = pred1_v_s3.sqrt()

        #mindisparity_s2 = pred2_s3_cur - (self.gamma_s3 + 1) * pred2_v_s4 - self.beta_s3
        #maxdisparity_s2 = pred2_s3_cur + (self.gamma_s3 + 1) * pred2_v_s4 + self.beta_s3
        mindisparity_s2 = pred1_s3_cur - pred1_v_s3
        maxdisparity_s2 = pred1_s3_cur + pred1_v_s3

        maxdisparity_s2 = F.upsample(maxdisparity_s2 * 2, [512 // 2, 1024 // 2], mode='bilinear', align_corners=True)  # left.size()
        mindisparity_s2 = F.upsample(mindisparity_s2 * 2, [512 // 2, 1024 // 2], mode='bilinear', align_corners=True)  # left.size()
        mindisparity_s2_1, maxdisparity_s2_1 = self.generate_search_range(self.sample_count_s2 + 1, mindisparity_s2, maxdisparity_s2, scale=1)


        disparity_samples_s2 = self.generate_disparity_samples(mindisparity_s2_1, maxdisparity_s2_1, self.sample_count_s2)
        
        warped_right_feature_map_left3, gather_index3, left_feature_map3, right_feature_map3 = self.spatial_pre(left_y_coordinate, features_left["concat_feature2"], features_right["concat_feature2"], disparity_samples_s2, 512)
        warped_right_feature_map_left4, gather_index4, left_feature_map4, right_feature_map4 = self.spatial_pre(left_y_coordinate, features_left["gw2"], features_right["gw2"], disparity_samples_s2, 512)
        
        #torch.Size([1, 12, 256, 512]) --> warped_right_feature_map_left3
        ##torch.Size([1, 6, 12, 256, 512])  --> gather_index3
        #torch.Size([1, 6, 12, 256, 512])  --> left_feature_map3
        ##torch.Size([1, 6, 12, 256, 512])  --> right_feature_map3
        #torch.Size([1, 12, 256, 512])  --> warped_right_feature_map_left4
        ##torch.Size([1, 80, 12, 256, 512])  --> gather_index4
        #torch.Size([1, 80, 12, 256, 512])  --> left_feature_map4
        ##torch.Size([1, 80, 12, 256, 512])  --> right_feature_map4
        #torch.Size([1, 12, 256, 512])  --> disparity_samples_s2

        return warped_right_feature_map_left3, gather_index3, left_feature_map3, right_feature_map3, warped_right_feature_map_left4, gather_index4, left_feature_map4, right_feature_map4, \
                disparity_samples_s2

    def forward3(self, *input):
        # =============================================================================================================================================
        # input: warped_right_feature_map_left3, gather_index3, left_feature_map3, right_feature_map3, warped_right_feature_map_left4, gather_index4, left_feature_map4, right_feature_map4, \
        #        disparity_samples_s2, feature_sparse['s2'], sparse_out["sparse2"], sparse_mask_out["sparse_mask2"]
        # output: pred1_s2

        feature_sparse = {}
        sparse_out = {}
        sparse_mask_out = {}
        warped_right_feature_map3, warped_right_feature_map4, warped_right_feature_map_left3, left_feature_map3, warped_right_feature_map_left4, left_feature_map4, \
                disparity_samples_s2, feature_sparse['s2'], sparse_out["sparse2"], sparse_mask_out["sparse_mask2"],  = input

        #warped_right_feature_map3 = torch.gather(right_feature_map.float(), dim=4, index=gather_index)
        #warped_right_feature_map4 = torch.gather(right_feature_map2.float(), dim=4, index=gather_index2)
        right_feature_map3 = warped_right_feature_map_left3 * warped_right_feature_map3  # [1，12，16，128，256]
        right_feature_map4 = warped_right_feature_map_left4 * warped_right_feature_map4
        confidence_v_concat_s2 = torch.cat((left_feature_map3, right_feature_map3), dim=1)
        confidence_v_gwc_s2 = groupwise_correlation_4D(left_feature_map4, right_feature_map4, self.num_groups)

        #confidence_v_concat_s2, _ = self.cost_volume_generator(features_left["concat_feature2"], features_right["concat_feature2"], disparity_samples_s2, 'concat')
        #confidence_v_gwc_s2, disparity_samples_s2 = self.cost_volume_generator(features_left["gw2"], features_right["gw2"], disparity_samples_s2, 'gwc', self.num_groups // 2)
        
        disparity_samples_s2 = disparity_samples_s2.unsqueeze(1)  # .float()
        confidence_v_s2 = torch.cat((confidence_v_gwc_s2, confidence_v_concat_s2, disparity_samples_s2), dim=1)
        # add gaussian modulation
        if self.sparse_feature_on:
            confidence_v_s2_s = torch.cat((confidence_v_s2, feature_sparse['s2'].repeat(1,1,confidence_v_s2.size()[2],1,1)), 1)
        else:
            confidence_v_s2_s = confidence_v_s2
        disparity_samples_s2 = torch.squeeze(disparity_samples_s2, dim=1)
        confidence_v_s2_m = self.cost_volum_modulation(sparse_out["sparse2"], sparse_mask_out["sparse_mask2"], disparity_samples_s2, confidence_v_s2_s)

        cost0_s2 = self.confidence0_s2(confidence_v_s2_m)
        # to here
        cost0_s2 = self.confidence1_s2(cost0_s2)  + cost0_s2  # torch.Size([1, 16, 12, 256, 512])
        out1_s2 = self.confidence2_s2(cost0_s2)
        out2_s2 = self.confidence3_s2(out1_s2)
        cost1_s2 = self.confidence_classif1_s2(out2_s2).squeeze(1)
        cost1_s2_possibility = F.softmax(cost1_s2, dim=1)
        pred1_s2 = torch.sum(cost1_s2_possibility * disparity_samples_s2, dim=1, keepdim=True)

        return pred1_s2

        res = [pred1_s2]
        res += [disparity_samples_s2, disparity_samples_s3]
        for item in [features_left, features_right]:
            for k in item:
                res.append(item[k])
                print(k)
        for item in res:
            print(item.shape)
        return tuple(res)

        # pred1_v_s2 = disparity_variance_confidence(cost1_s2_possibility, disparity_samples_s2, pred1_s2)
        # pred1_v_s2 = pred1_v_s2.sqrt()
        # ValueError: align_corners option can only be set with the interpolating modes: linear | bilinear | bicubic | trilinear
        # print([4*left.size()[2], 2*left.size()[3]], pred2_s4.shape)  # [2048, 2048] torch.Size([1, 1, 64, 128])
        # pred2_s4 = F.upsample(pred2_s4 * 16, [4*left.size()[2], 2*left.size()[3]], mode='bilinear', align_corners=True)
        # pred2_s4 = torch.squeeze(pred2_s4, 1)
        # print([4*left.size()[2], 2*left.size()[3]], pred1_s3.shape)  # [2048, 2048] torch.Size([1, 1, 128, 256])
        # pred1_s3_up = F.upsample(pred1_s3 * 8, [4*left.size()[2], 2*left.size()[3]], mode='bilinear', align_corners=True)
        # pred1_s3_up = torch.squeeze(pred1_s3_up, 1)
        # print([4*left.size()[2], 2*left.size()[3]], pred1_s2.cpu().shape)  # [2048, 2048] torch.Size([1, 1, 256, 512])
        pred1_s2 = F.upsample(pred1_s2 * 4, [4*left.size()[2], 2*left.size()[3]], mode='bilinear', align_corners=True)
        pred1_s2 = torch.squeeze(pred1_s2, 1)

        print('net done')
        return pred1_s2
        #return (pred1_s2, pred1_s3_up, pred2_s4)


def CFNet(d):
    return cfnet(d, use_concat_volume=True)
