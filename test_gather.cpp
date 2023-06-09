#include <sys/time.h>
#include "glog/logging.h"
#include <cnrt.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cnml.h>
#include <assert.h>
#include "MYGATHER/Mygather.h"

void readData(float* data, int length, std::string filename) {
    std::ifstream file;
    file.open(filename);
    float temp = 0.0;
    int j = 0;
    while (file >> temp) {
          data[j] = temp;
          j++;
    }
    file.close();
    CHECK_EQ(j, length)<< "plase check the input data dim";
}


void * float32_2_float16_and_NCHW_2_NHWC_MLU(float * src, cnrtDataType_t src_type, int src_size,
                                          cnrtDataType_t dst_type, int dimNum, int dimValue[], int dimOrder[]){
  void * dst = reinterpret_cast<void*>(malloc(cnrtDataTypeSize(dst_type) * src_size));
  cnrtRet_t ret = cnrtTransOrderAndCast(reinterpret_cast<void*>(src), src_type, dst, dst_type, NULL,dimNum, dimValue, dimOrder);
  assert(ret == CNRT_RET_SUCCESS);
  void * dst_mlu = NULL;
  cnrtMalloc(&dst_mlu, cnrtDataTypeSize(dst_type) * src_size);
  cnrtMemcpy(dst_mlu, dst, cnrtDataTypeSize(dst_type) * src_size, CNRT_MEM_TRANS_DIR_HOST2DEV);
  free(dst);
  return dst_mlu;
                                          }

void * float16_2_float32_and_NHWC_MLU_2_NCHW_CPU(void * src, cnrtDataType_t src_type, int src_size,
                                         cnrtDataType_t dst_type, int dimNum, int dimValue[], int dimOrder[]){
  void * dst = reinterpret_cast<void*>(malloc(cnrtDataTypeSize(src_type) * src_size));
  cnrtMemcpy(dst,src,cnrtDataTypeSize(src_type) * src_size,CNRT_MEM_TRANS_DIR_DEV2HOST);
  
  void * dst_cpu = reinterpret_cast<void*>(malloc(cnrtDataTypeSize(dst_type) * src_size));
  cnrtRet_t ret = cnrtTransOrderAndCast(dst, src_type, dst_cpu, dst_type, NULL,dimNum, dimValue, dimOrder);
  assert(ret == CNRT_RET_SUCCESS);
  return dst_cpu;
}


int main(int argc, char* argv[]) {
  cnrtInit(0);
  unsigned devNum;
  cnrtGetDeviceCount(&devNum);
  LOG(INFO)<<"there is "<<devNum<<" MLU device";
  cnrtDev_t dev;
  int Dev_use = 0;
  cnrtGetDeviceHandle(&dev, Dev_use); 
  cnrtSetCurrentDevice(dev);

  // 读取2个数据文件，分别为forward[1] -> index
  // forward[3] -> data
  int index_size = 1*16*128*256*12; // NHWC
  int data_size = 1*16*128*256*12;
  auto databuf_index = reinterpret_cast<float*>(malloc(sizeof(float) * index_size)); 
  auto databuf_data = reinterpret_cast<float*>(malloc(sizeof(float) * data_size)); 
  
  // N C H W
  readData(databuf_index, index_size, "../origin_gather_input/gather_input[1].txt"); 
  readData(databuf_data, data_size, "../origin_gather_input/gather_input[0].txt");

  // N C H W -> N H W C
  // CNRTFLOAT32 -> CNRTFLOAT16
  // cnrtTransOrderAndCast
  int shape_index[4] = {12,16,128,256};
  // int shape_index[4] = {1, 1, 1, 256};
  // int order_index[4] = {0,2,3,1};
  int order_index[4] = {0,1,2,3};
  void * index_mlu = float32_2_float16_and_NCHW_2_NHWC_MLU(databuf_index, CNRT_FLOAT32, index_size, CNRT_FLOAT16, 4, shape_index,order_index);

  int shape_data[4] = {12,16,128,256};
  // int order_data[4] = {0,2,3,1};
  int order_data[4] = {0,1,2,3};
  void * data_mlu = float32_2_float16_and_NCHW_2_NHWC_MLU(databuf_data, CNRT_FLOAT32, data_size, CNRT_FLOAT16, 4, shape_data, order_data);

  cnrtQueue_t cnrt_queue;
  cnrtCreateQueue(&cnrt_queue);
  void * gather_output_mlu = NULL;
  gather_output_mlu = gather_mlu(data_mlu, shape_data, index_mlu, shape_index, CNML_DIM_C , cnrt_queue);

  // int shape_output[4] = {16,128,256,12}; // NHWC
  // int order_output[4] = {0,3,1,2}; // NCHW
  int shape_output[4] = {12,16,128,256}; // NHWC
  int order_output[4] = {0,1,2,3}; // NCHW
  void * res = float16_2_float32_and_NHWC_MLU_2_NCHW_CPU(gather_output_mlu, CNRT_FLOAT16, index_size, CNRT_FLOAT32, 4, shape_data, order_data);

  std::string output_path = "../mygather_output/";
  std::stringstream ss;
  ss << output_path  << "gather_output" ;
  std::string output_name = ss.str();
  LOG(INFO)<<"writing output to "<<output_name;
  std::ofstream fout(output_name, std::ios::out);
  fout << std::flush;
  for(int j = 0 ; j < index_size; j++){
    fout << ((float*)res)[j] << std::endl;
  }
  fout<<std::flush;
  fout.close();

  LOG(INFO)<<"test Done.";
  return 0;
}

