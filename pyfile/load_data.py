import torch
import numpy as np

def load_data(filename):
    if filename == "forward1_res.pt" :
        data = torch.load(f"../data_6_5/{filename}")
        print(len(data))
        for i,d in enumerate(data) :
            print(d.shape)
            # if len(d.shape) == 5:
            #     # B N C H W -> B N H W C
            #     d = d.permute(0,2,3,4,1)
            # elif len(d.shape) ==4:
            #     d = d.permute(0,2,3,1)
            # print(d.shape)
            d = d.flatten(start_dim=0)
            np.savetxt(f"../data_6_5/forward1_res[{i}].txt",d.numpy(),fmt="%.7f")
    
    if filename == "warped_right_feature_map.pt":
        data = torch.load(f"../data_6_5/{filename}")
        print(len(data))
        for i,d in enumerate(data) :
            print(d.shape)
            d = d.flatten(start_dim=0)
            np.savetxt(f"../data/Cfnet1_gather_[{i}].txt",d.numpy(),fmt="%.7f")
    
    if filename == "forward3_res.pt":
        data = torch.load(f"../gather_output/forward3_res.pt")
        print(len(data))
        for i , d in enumerate(data):
            print(d.shape)
            d = d.flatten(start_dim=0)
            np.savetxt(f"../gather_output/forward3_res[{i}].txt",d.numpy(),fmt="%.7f")

if __name__ =='__main__':
    # load_data("forward1_res.pt")
    load_data("forward3_res.pt")