import numpy as np
from tqdm import tqdm
import logging

# input_file0_o = "forward1_res[3].txt"
# input_file0_m = "cfnet1.cambricon_output_3"

# input_file1_o = "forward1_res[1].txt"
# input_file1_m = "cfnet1.cambricon_output_1"

# file1 = "Cfnet1_gather_[0].txt"
# file2 = "cfnet1.cambricongather_output"

filelist = [] 
test_file = [8, 10, 11, 12, 13, 14, 15, 17, 18]
for id in test_file:
    filelist.append(["forward1_res["+str(id)+"].txt", "cfnet1.cambricon_output_"+str(id)])
logging.basicConfig(level=logging.INFO)
count =  0 ;
# filelist = [[input_file0_o,input_file0_m],[input_file1_o,input_file1_m],[file1,file2]]
for f in tqdm(filelist):
    data1 = np.loadtxt("data_6_5/"+f[0])
    data2 = np.loadtxt("output/"+f[1])
    logging.info("comparing " + f[0] + " and " + f[1])
    assert data1.size == data2.size, "size not equal."
    if(np.allclose(data1, data2)):
        print("all close pass.")
        count+=1
    else:
        print("all close failed.")
logging.info(f"pass rate is {count/len(filelist)}")
