import torch
import numpy as np

def load_data(filename):
    data = torch.load(f"../data/{filename}")
    print(len(data))
    if filename == "forward1_res.pt" :
        for i,d in enumerate(data) :
            print(d.shape)
            # if len(d.shape) == 5:
            #     # B N C H W -> B N H W C
            #     d = d.permute(0,2,3,4,1)
            # elif len(d.shape) ==4:
            #     d = d.permute(0,2,3,1)
            # print(d.shape)
            d = d.flatten(start_dim=0)
            np.savetxt(f"../data/forward1_res[{i}].txt",d.numpy(),fmt="%.7f")
    
    if filename == "warped_right_feature_map.pt":
        for i,d in enumerate(data) :
            print(d.shape)
            d = d.flatten(start_dim=0)
            np.savetxt(f"../data/Cfnet1_gather_[{i}].txt",d.numpy(),fmt="%.7f")

if __name__ =='__main__':
    load_data("forward1_res.pt")