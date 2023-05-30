from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
# from datasets import __datasets__
from __init__ import __models__, model_loss
# from utils import *
from torch.utils.data import DataLoader
# from datasets import listfiles as ls
# from datasets import eth3dLoader as DA
# from datasets import MiddleburyLoader as mid
import gc
import torch_mlu
import torch_mlu.core.mlu_model as ct
import torch_mlu.core.mlu_model as mm
import torch_mlu.core.mlu_quantize as mlu_quantize
torch.set_grad_enabled(False)
ct.set_core_number(16)
ct.set_core_version("MLU270")
os.environ['TFU_ENABLE'] = '1'
os.environ['ATEN_CNML_COREVERSION'] = 'MLU200'
os.environ['CNML_ADDR_OPT'] = 'false'


# 获取网络
model = __models__["cfnet"](int("512"))
model = nn.DataParallel(model)
quantized_net = torch_mlu.core.mlu_quantize.quantize_dynamic_mlu(model)
# state_dict = torch.load('./finetune_int16_v5.0.pt')
state_dict = torch.load('./finetune_int8_v2.0.pt')
quantized_net.load_state_dict(state_dict, strict=False)
quantized_net.eval()
device = ct.mlu_device()
quantized_net.to(device)
# # 配置量化参数
# qconfig={'iteration': 1, 'use_avg':False, 'data_scale':1.0, 'firstconv':False, 'per_channel': False}
# # 调用量化接口
# quantized_net = mlu_quantize.quantize_dynamic_mlu(model.float(),qconfig_spec=qconfig, dtype='int16', gen_quant=True)
# # 设置为推理模式
# quantized_net = quantized_net.eval().float()
# # 执行推理生成量化值 
# imgL = torch.load("/stereo_lidar_downsample/imgl_data.pt")
# imgR = torch.load("/stereo_lidar_downsample/imgR_data.pt")
# disp_sparse = torch.load("/stereo_lidar_downsample/disp_sparse_data.pt")
# sparse_mask = torch.load("/stereo_lidar_downsample/sparse_mask_data.pt")
# quantized_net(imgL, imgR, disp_sparse, sparse_mask)
# # 保存量化模型
# torch.save(quantized_net.state_dict(), './finetune_int16_v1.0.pt')


trace_input1 = tuple([
    torch.randn(1, 3, 512, 1024, dtype=torch.float).half().to(ct.mlu_device()), 
    torch.randn(1, 3, 512, 1024, dtype=torch.float).half().to(ct.mlu_device()),
    torch.randn(1, 1, 512, 1024, dtype=torch.float).half().to(ct.mlu_device()),
    torch.randn(1, 1, 512, 1024, dtype=torch.float).half().to(ct.mlu_device()),
    torch.linspace(0.0, 255, 256).half().to(ct.mlu_device())
    ])


trace_input2 = tuple([
    torch.randn([1, 12, 16, 128, 256]).half().to(ct.mlu_device()),#  warped_right_feature_map
    torch.randn([1, 16, 128, 256]).half().to(ct.mlu_device()),
    torch.randn([1, 12, 16, 128, 256]).half().to(ct.mlu_device()),
    torch.randn([1, 160, 16, 128, 256]).half().to(ct.mlu_device()),#  warped_right_feature_map_left2
    torch.randn([1, 16, 128, 256]).half().to(ct.mlu_device()),
    torch.randn([1, 160, 16, 128, 256]).half().to(ct.mlu_device()),
    torch.randn([1, 16, 128, 256]).half().to(ct.mlu_device()),
    torch.randn([1, 1, 1, 128, 256]).half().to(ct.mlu_device()),
    torch.randn([1, 1, 128, 256]).half().to(ct.mlu_device()),
    torch.randn([1, 1, 128, 256]).half().to(ct.mlu_device()),
    torch.randn([1, 6, 256, 512]).half().to(ct.mlu_device()),
    torch.randn([1, 6, 256, 512]).half().to(ct.mlu_device()),
    torch.randn([1, 80, 256, 512]).half().to(ct.mlu_device()),
    torch.randn([1, 80, 256, 512]).half().to(ct.mlu_device()),
    torch.linspace(0.0, 511, 512).half().to(ct.mlu_device())
])

trace_input3 = tuple([
    torch.randn([1, 6, 12, 256, 512]).half().to(ct.mlu_device()),
    torch.randn([1, 80, 12, 256, 512]).half().to(ct.mlu_device()),
    torch.randn([1, 12, 256, 512]).half().to(ct.mlu_device()),
    #torch.randn([1, 6, 12, 256, 512]).half().to(ct.mlu_device()),
    torch.randn([1, 6, 12, 256, 512]).half().to(ct.mlu_device()),
    #torch.randn([1, 6, 12, 256, 512]).half().to(ct.mlu_device()),
    torch.randn([1, 12, 256, 512]).half().to(ct.mlu_device()),
    #torch.randn([1, 80, 12, 256, 512]).half().to(ct.mlu_device()),
    torch.randn([1, 80, 12, 256, 512]).half().to(ct.mlu_device()),
    #torch.randn([1, 80, 12, 256, 512]).half().to(ct.mlu_device()),
    torch.randn([1, 12, 256, 512]).half().to(ct.mlu_device()),
    torch.randn([1, 1, 1, 256, 512]).half().to(ct.mlu_device()),
    torch.randn([1, 1, 256, 512]).half().to(ct.mlu_device()),
    torch.randn([1, 1, 256, 512]).half().to(ct.mlu_device())
])

trace_input = trace_input1
ct.save_as_cambricon('cfnet1')
with torch.no_grad():
    quantized_net = torch.jit.trace(quantized_net, trace_input, check_trace = False)


print("==================inference=======================")
out_put  = quantized_net(*trace_input)
print(len(out_put))
for item in out_put:
    print(item.cpu().shape)
