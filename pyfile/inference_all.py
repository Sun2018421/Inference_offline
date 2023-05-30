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
model = torch_mlu.core.mlu_quantize.quantize_dynamic_mlu(nn.DataParallel(model))
#quantized_net = torch_mlu.core.mlu_quantize.quantize_dynamic_mlu(model)
# state_dict = torch.load('./finetune_int16_v5.0.pt')
state_dict = torch.load('./finetune_int8_v2.0.pt')
model.load_state_dict(state_dict, strict=False)
model.eval().to(ct.mlu_device())
#device = ct.mlu_device()
#quantized_net.to(device)
# # 配置量化参数
# qconfig={'iteration': 1, 'use_avg':False, 'data_scale':1.0, 'firstconv':False, 'per_channel': False}
# # 调用量化接口
# quantized_net = mlu_quantize.quantize_dynamic_mlu(model.float(),qconfig_spec=qconfig, dtype='int16', gen_quant=True)
# # 设置为推理模式
# quantized_net = quantized_net.eval().float()
# # 执行推理生成量化值 
imgL = torch.load("./imgl_data.pt")
imgR = torch.load("./imgR_data.pt")
disp_sparse = torch.load("./disp_sparse_data.pt")
sparse_mask = torch.load("./sparse_mask_data.pt")
# quantized_net(imgL, imgR, disp_sparse, sparse_mask)
# # 保存量化模型
# torch.save(quantized_net.state_dict(), './finetune_int16_v1.0.pt')


trace_input = [
    imgL, 
    imgR,
    disp_sparse,
    sparse_mask,
    torch.linspace(0.0, 255, 256)
]

for i in range(len(trace_input)):
    trace_input[i] = trace_input[i].half().to(ct.mlu_device())
trace_input = tuple(trace_input)
#ct.save_as_cambricon('cfnet1')
#with torch.no_grad():
#    quantized_net = torch.jit.trace(quantized_net, trace_input, check_trace = False)


print("==================inference=======================")

feature_sparse = {}
sparse_out = {}
sparse_mask_out = {}
features_left = {}
features_right = {}

warped_right_feature_map_left, gather_index, left_feature_map, right_feature_map, warped_right_feature_map_left2, \
gather_index2, left_feature_map2, right_feature_map2, \
disparity_samples_s3, feature_sparse['s3'], sparse_out["sparse3"], sparse_mask_out["sparse_mask3"], \
features_left["concat_feature2"], features_right["concat_feature2"], features_left["gw2"], features_right["gw2"],  \
feature_sparse['s2'], sparse_out["sparse2"], sparse_mask_out["sparse_mask2"] = model(*trace_input)

warped_right_feature_map = torch.gather(right_feature_map.cpu().float(), dim=4, index=gather_index.cpu().long()).half().to(ct.mlu_device())
warped_right_feature_map2 = torch.gather(right_feature_map2.cpu().float(), dim=4, index=gather_index2.cpu().long()).half().to(ct.mlu_device())

warped_right_feature_map_left3, gather_index3, left_feature_map3, right_feature_map3, warped_right_feature_map_left4,\
gather_index4, left_feature_map4, right_feature_map4, disparity_samples_s2 = \
model(warped_right_feature_map, warped_right_feature_map_left, left_feature_map,
        warped_right_feature_map2, warped_right_feature_map_left2,  left_feature_map2,  \
disparity_samples_s3, feature_sparse['s3'], sparse_out["sparse3"], sparse_mask_out["sparse_mask3"], \
features_left["concat_feature2"], features_right["concat_feature2"], features_left["gw2"], features_right["gw2"],\
      torch.linspace(0.0, 511, 512).half().to(ct.mlu_device()))

warped_right_feature_map3 = torch.gather(right_feature_map3.cpu().float(), dim=4, index=gather_index3.cpu().long()).half().to(ct.mlu_device())
warped_right_feature_map4 = torch.gather(right_feature_map4.cpu().float(), dim=4, index=gather_index4.cpu().long()).half().to(ct.mlu_device())

output3 = model(warped_right_feature_map3, warped_right_feature_map4, warped_right_feature_map_left3, left_feature_map3,\
                 warped_right_feature_map_left4, left_feature_map4, \
disparity_samples_s2, feature_sparse['s2'], sparse_out["sparse2"], sparse_mask_out["sparse_mask2"])

