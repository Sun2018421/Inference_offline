#include <sys/time.h>
#include "glog/logging.h"
#ifdef USE_MLU
#include <cnrt.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cnml.h>
#include <assert.h>

void linspace(float start, float stop, int num,float * data){
  float step = (stop - start)/(num - 1);
  for (int i = 0 ; i < num ; ++i){
    data[i] = start + step * i ;
  }
}

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

void * gather_mlu(void * & input_tensor, int * input_shape, void * &index_tensor, int * index_shape , cnmlDimension_t dim, cnrtQueue_t &cnrt_queue){
    cnmlBaseOp_t cast_op = NULL;
    cnmlTensor_t cast_input = NULL;
    cnmlCreateTensor_V2(&cast_input, CNML_TENSOR);
    cnmlSetTensorShape_V2(cast_input, 4, index_shape, NULL);
    cnmlSetTensorDataType(cast_input, CNML_DATA_FLOAT16);

    cnmlTensor_t cast_output = NULL;
    cnmlCreateTensor_V2(&cast_output, CNML_TENSOR);
    cnmlSetTensorShape_V2(cast_output, 4, index_shape, NULL);
    cnmlSetTensorDataType(cast_output, CNML_DATA_INT32);

    cnmlCreateCastOp(&cast_op, CNML_CAST_FLOAT16_TO_INT32_ROUND_ZERO, cast_input, cast_output);
    cnmlCompileBaseOp_V2(cast_op);
    auto cast_input_mlu_ptr = index_tensor;
    void * cast_output_mlu = NULL;
    cnrtRet_t res = cnrtMalloc(&cast_output_mlu, index_shape[0]*index_shape[1]*index_shape[2]*index_shape[3]*sizeof(int));
    assert(res == CNRT_RET_SUCCESS);
    cnmlStatus_t cast_ret =  cnmlComputeCastOpForward_V4(cast_op, NULL, cast_input_mlu_ptr, NULL, cast_output_mlu, cnrt_queue, NULL);
    if(cast_ret != CNML_STATUS_SUCCESS){
      LOG(FATAL)<<"Cast Failed and ret is" <<cast_ret ;
    }
    if (cnrtSyncQueue(cnrt_queue) == CNRT_RET_SUCCESS) {
      // LOG(INFO) << "Cast Done!";
    } else {
      LOG(FATAL) << "SyncQueue Error ";
    }
    
    cnmlTensor_t gather_input_tensor = NULL;
    cnmlCreateTensor_V2(&gather_input_tensor, CNML_TENSOR);    
    cnmlSetTensorShape_V2(gather_input_tensor, 4, input_shape,NULL);
    cnmlSetTensorDataType(gather_input_tensor, CNML_DATA_FLOAT16); 

    cnmlTensor_t gather_index_tensor = NULL;
    cnmlCreateTensor_V2(&gather_index_tensor, CNML_TENSOR);
    cnmlSetTensorShape_V2(gather_index_tensor, 4, index_shape,NULL);
    cnmlSetTensorDataType(gather_index_tensor, CNML_DATA_INT32);

    cnmlTensor_t gather_output_tensor = NULL;
    cnmlCreateTensor_V2(&gather_output_tensor, CNML_TENSOR);
    cnmlSetTensorShape_V2(gather_output_tensor, 4, index_shape, NULL);
    cnmlSetTensorDataType(gather_output_tensor, CNML_DATA_FLOAT16); 

    cnmlBaseOp_t gather_op = NULL;
    cnmlCreateGatherV2Op(&gather_op, gather_input_tensor, gather_index_tensor, gather_output_tensor, dim);
    cnmlCompileBaseOp_V2(gather_op);

    void * gather_output_mlu_ptr  = NULL;
    res = cnrtMalloc(&gather_output_mlu_ptr, index_shape[0]*index_shape[1]*index_shape[2]*index_shape[3]*sizeof(float)); // /2去掉了，是不是开小了？
    assert(res == CNRT_RET_SUCCESS);
    // LOG(INFO)<<"start gather"<<" "<<"shape is "<<index_shape[0]<<" "<<index_shape[1]<<" "<<index_shape[2]<<" "<<index_shape[3];
    cnmlStatus_t compute_ret = cnmlComputeGatherV2OpForward_V4(gather_op, NULL, input_tensor, NULL, cast_output_mlu, NULL, gather_output_mlu_ptr, cnrt_queue, NULL);
    if(compute_ret != CNML_STATUS_SUCCESS){
      LOG(FATAL)<<"CompuateGatherV2 Failed and ret is" <<compute_ret ;
    }
    if (cnrtSyncQueue(cnrt_queue) == CNRT_RET_SUCCESS) {
      // LOG(INFO) << "Gather Done!" ;
    } else {
      LOG(FATAL) << "SyncQueue Error ";
    }
    cnrtFree(cast_output_mlu);
    cnmlDestroyBaseOp(&cast_op);
    cnmlDestroyBaseOp(&gather_op);

    return gather_output_mlu_ptr;
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
  cnrtModel_t model;
  std::string fname = (std::string)"cfnet1.cambricon";
  int size;
  cnrtGetModelSize(fname.c_str(), &size);
  cnrtLoadModel(&model, fname.c_str());
  cnrtFunction_t function;
  cnrtRuntimeContext_t rt_ctx_;

  struct timeval tpend, tpstart;
  gettimeofday(&tpstart, NULL);
  std::string name = (std::string)"subnet0";
  cnrtCreateFunction(&function);
  assert(cnrtExtractFunction(&function, model, name.c_str())==CNRT_RET_SUCCESS);
  cnrtCreateRuntimeContext(&rt_ctx_, function, NULL);
  int inputNum, outputNum;
  int64_t* inputSizeS = nullptr;
  int64_t* outputSizeS = nullptr;
  cnrtGetInputDataSize(&inputSizeS, &inputNum, function);
  cnrtGetOutputDataSize(&outputSizeS, &outputNum, function);
  cnrtDataType_t* input_data_type = nullptr;
  cnrtDataType_t* output_data_type = nullptr;
  cnrtGetInputDataType(&input_data_type, &inputNum, function);
  cnrtGetOutputDataType(&output_data_type, &outputNum, function);

  void** inputCpuPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * inputNum));
  void** param = reinterpret_cast<void**>(malloc(sizeof(void*) * (inputNum + outputNum)));

  for (int i = 0; i < inputNum; i++) {
    int ip = inputSizeS[i] / cnrtDataTypeSize(input_data_type[i]); 
    LOG(INFO)<<"the "<<i<<" input elements is "<<ip;
    auto databuf = reinterpret_cast<float*>(malloc(sizeof(float) * ip)); 
    switch(i){
      case 1:
        readData(databuf,ip,"../data/imgR_data.txt");
      break;
      case 4:
        readData(databuf,ip,"../data/imgl_data.txt");
      break;
      case 3:
        readData(databuf,ip,"../data/disp_sparse_data.txt");
      break;
      case 2:
        readData(databuf,ip,"../data/sparse_mask_data.txt");
      break;
      case 0:
        linspace(0.0,255,256,databuf);
      break;
    }
    inputCpuPtrS[i] = reinterpret_cast<void*>(databuf);  // NCHW
  }

  void** inputMluPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * inputNum));
  void** outputMluPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * outputNum));

  for (int i = 0; i < inputNum; i++) {
    cnrtMalloc(&inputMluPtrS[i], inputSizeS[i]);
  }
  for (int i = 0; i < outputNum; i++) {
    cnrtMalloc(&outputMluPtrS[i], outputSizeS[i]);
  }
  for (int i = 0; i < inputNum; i++) {
    param[i] = inputMluPtrS[i];
  }
  for (int i = 0; i < outputNum; i++) {
    param[inputNum + i] = outputMluPtrS[i];
  }
  // 更换维度
  void** tempPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * inputNum));
  void* temp_input_cpu_data = nullptr;
  for (int i = 0; i < inputNum; i++) {
    int ip = inputSizeS[i] / cnrtDataTypeSize(input_data_type[i]);
    auto databuf = reinterpret_cast<float*>(malloc(sizeof(float) * ip));
    tempPtrS[i] = reinterpret_cast<void*>(databuf);
    std::vector<int> shape(4, 1);
    int dimNum = 4;
    cnrtGetInputDataShape((int**)&shape, &dimNum, i, function);
    // 输出每个输入的形状
    LOG(INFO)<<"the shape of input "<<i<<" is "<<shape[0]<<" "<<shape[1]<<" "<<shape[2]<<" "<<shape[3];    
    int dim_order[4] = {0, 2, 3, 1};
    int dim_shape[4] = {shape[0], shape[3],
                        shape[1], shape[2]};  // NCHW
    cnrtTransDataOrder(inputCpuPtrS[i], CNRT_FLOAT32, tempPtrS[i], //TODO: float32->float16
                        4, dim_shape, dim_order);
    temp_input_cpu_data = (void*)malloc(inputSizeS[i]);
    int input_count = inputSizeS[i] / cnrtDataTypeSize(input_data_type[i]);
    if (input_data_type[i] != CNRT_FLOAT32) {
      cnrtCastDataType(tempPtrS[i],
                        CNRT_FLOAT32,
                        temp_input_cpu_data,
                        input_data_type[i],
                        input_count,
                        nullptr);
    } else {
      temp_input_cpu_data = tempPtrS[i];
    }
    cnrtMemcpy(inputMluPtrS[i],
                temp_input_cpu_data,
                inputSizeS[i],
                CNRT_MEM_TRANS_DIR_HOST2DEV);

    if (temp_input_cpu_data) {
      free(temp_input_cpu_data);
      temp_input_cpu_data = nullptr;
    }
  }

  cnrtQueue_t cnrt_queue;
  cnrtCreateQueue(&cnrt_queue);
  cnrtSetRuntimeContextDeviceId(rt_ctx_, Dev_use);
  cnrtInitRuntimeContext(rt_ctx_, NULL);
  cnrtNotifier_t notifierBeginning, notifierEnd;
  cnrtCreateNotifier(&notifierBeginning);
  cnrtCreateNotifier(&notifierEnd);
  float event_time_use;
  // run MLU
  // place start_event to cnrt_queue
  cnrtPlaceNotifier(notifierBeginning, cnrt_queue);
  CNRT_CHECK(cnrtInvokeRuntimeContext(rt_ctx_, param, cnrt_queue, nullptr));
  // place end_event to cnrt_queue
  cnrtPlaceNotifier(notifierEnd, cnrt_queue);
  if (cnrtSyncQueue(cnrt_queue) == CNRT_RET_SUCCESS) {
    // get start_event and end_event elapsed time
    cnrtNotifierDuration(notifierBeginning, notifierEnd, &event_time_use);
    // LOG(INFO) << " hardware time: " << event_time_use;
  } else {
    LOG(INFO) << " SyncQueue Error ";
  }

  for (int i = 0; i < inputNum; i++) {
    cnrtFree(inputMluPtrS[i]);
  }

  cnrtDestroyNotifier(&notifierBeginning);
  cnrtDestroyNotifier(&notifierEnd);
  free(inputCpuPtrS);
  
  gettimeofday(&tpend,NULL);
  float execTime = 1000000 * (tpend.tv_sec - tpstart.tv_sec) +
    tpend.tv_usec - tpstart.tv_usec;
  LOG(INFO) << " cfnet1 time: " << execTime << " us";

/* 将模型输出保存出来，改变数据布局，NHWC->NCHW
*/

  std::string output_path = "../output/";
  void ** outputCpuPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * outputNum));
  void ** outPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * outputNum));
  void * temp_output_cpu_data = nullptr;
  for(int i = 0; i < outputNum ; i++){
    temp_output_cpu_data = (void*)malloc(outputSizeS[i]);
    cnrtMemcpy(temp_output_cpu_data,
              outputMluPtrS[i],
              outputSizeS[i],
              CNRT_MEM_TRANS_DIR_DEV2HOST);
    int output_count = outputSizeS[i] / cnrtDataTypeSize(output_data_type[i]);
    outputCpuPtrS[i] = reinterpret_cast<void*>(reinterpret_cast<float*>(malloc(sizeof(float) * output_count)));
    if (output_data_type[i] != CNRT_FLOAT32) {
      cnrtCastDataType(temp_output_cpu_data,
                        output_data_type[i],
                        outputCpuPtrS[i],
                        CNRT_FLOAT32,
                        output_count,
                        nullptr);
    } else {
      memcpy(outputCpuPtrS[i], temp_output_cpu_data, outputSizeS[i]);
    }
    if(i == 8|| i==10|| i==11 || i==12 || i==13|| i==14|| i==15 || i==17 || i==18){
      outPtrS[i] = reinterpret_cast<void*>(reinterpret_cast<float*>(malloc(sizeof(float) * output_count)));
      std::vector<int> shape_for_output_cast(4, 1);
      int dimNum_for_output_cast = 4;
      cnrtGetOutputDataShape((int**)&shape_for_output_cast, &dimNum_for_output_cast, i, function);
      int dim_order_for_output_cast[4] = {0, 3, 1, 2};
      int dim_shape_for_output_cast[4] = {shape_for_output_cast[0], shape_for_output_cast[1],
                          shape_for_output_cast[2], shape_for_output_cast[3]};  // NHWC
                        
      cnrtRet_t ret = cnrtTransDataOrder(outputCpuPtrS[i], CNRT_FLOAT32, outPtrS[i],
                          4, dim_shape_for_output_cast, dim_order_for_output_cast);
      assert(ret == CNRT_RET_SUCCESS);
    }else{
      outPtrS[i] = reinterpret_cast<void*>(reinterpret_cast<float*>(malloc(sizeof(float) * output_count)));
      std::vector<int> shape_for_output_cast(5, 1);
      int dimNum_for_output_cast = 4;
      cnrtGetOutputDataShape((int**)&shape_for_output_cast, &dimNum_for_output_cast, i, function);
      int dim_order_for_output_cast[4] = {0, 3, 1, 2};
      int dim_shape_for_output_cast[4] = {shape_for_output_cast[1], shape_for_output_cast[2],
                          shape_for_output_cast[3], shape_for_output_cast[4]};  // NHWC
                        
      cnrtRet_t ret = cnrtTransDataOrder(outputCpuPtrS[i], CNRT_FLOAT32, outPtrS[i],
                          4, dim_shape_for_output_cast, dim_order_for_output_cast);
      assert(ret == CNRT_RET_SUCCESS);
    }
 
    std::stringstream ss;
    ss << output_path << fname << "_output_" << i;
    std::string output_name = ss.str();
    LOG(INFO)<<"writing output to "<<output_name;
    std::ofstream fout(output_name, std::ios::out);
    fout << std::flush;
    for(int j = 0 ; j < output_count; j++){
      fout << (reinterpret_cast<float*>(outPtrS[i]))[j] << std::endl;
    }
    fout<<std::flush;
    fout.close();
    
    free(outputCpuPtrS[i]);
    free(outPtrS[i]);
    free(temp_output_cpu_data);
    temp_output_cpu_data = nullptr;
  }
  free(outputCpuPtrS);

  gettimeofday(&tpstart,NULL);

  // for gather1: 因为dim=4, 去掉第一维并不影响数据排放 dim=3 -> NCHW
  std::vector<int> shape_input(5,1); int dimNum = 5;
  cnrtGetOutputDataShape((int **)&shape_input, &dimNum, 3, function);
  // LOG(INFO)<<"the shape of right_feature_map: "<<" "<<shape_input[1]<<" "<<shape_input[2]<<" "<<shape_input[3]<<" "<<shape_input[4];
  int input_shape[] = {shape_input[1], shape_input[2], shape_input[3], shape_input[4]};
  std::vector<int> shape_index(5,1); int dimNum_index = 5;
  cnrtGetOutputDataShape((int **)&shape_index, &dimNum_index, 1, function);
  int index_shape[] = {shape_index[1], shape_index[2], shape_index[3], shape_index[4]};
  void* gather_output_mlu1 = gather_mlu(outputMluPtrS[3], input_shape, outputMluPtrS[1],index_shape,CNML_DIM_C,cnrt_queue);

  int output_size = cnrtDataTypeSize(output_data_type[3]) * index_shape[0] * index_shape[1] * index_shape[2] * index_shape[3];
  void * temp_output_gather_cpu_data = (void *)malloc(output_size);
  cnrtMemcpy(temp_output_gather_cpu_data, 
    gather_output_mlu1, 
    output_size,
    CNRT_MEM_TRANS_DIR_DEV2HOST);

  int output_gather_count = output_size / cnrtDataTypeSize(output_data_type[3]);
  void * output_gather_cpu_data = (void *)malloc(sizeof(float) * output_gather_count);
  if (output_data_type[3] != CNRT_FLOAT32) {
    cnrtCastDataType(temp_output_gather_cpu_data,
                      output_data_type[3],
                      output_gather_cpu_data,
                      CNRT_FLOAT32,
                      output_gather_count,
                      nullptr);
  } else {
    memcpy(output_gather_cpu_data, temp_output_gather_cpu_data, output_size);
  }
  std::stringstream ss;
  ss << output_path << fname << "gather_output" ;
  std::string output_name = ss.str();
  LOG(INFO)<<"writing output to "<<output_name;
  std::ofstream fout(output_name, std::ios::out);
  fout << std::flush;
  for(int j = 0 ; j < output_gather_count; j++){
    fout << ((float*)output_gather_cpu_data)[j] << std::endl;
  }
  fout<<std::flush;
  fout.close();
  free(output_gather_cpu_data);
  free(temp_output_gather_cpu_data);
  temp_output_gather_cpu_data = nullptr;

  cnrtGetOutputDataShape((int **)&shape_input, &dimNum, 7, function);
  input_shape[0] = shape_input[1]; input_shape[1] = shape_input[2]; input_shape[2] = shape_input[3]; input_shape[3] = shape_input[4];
  cnrtGetOutputDataShape((int **)&shape_index, &dimNum_index, 5, function);
  index_shape[0] = shape_index[1]; index_shape[1] = shape_index[2]; index_shape[2] = shape_index[3]; index_shape[3] = shape_index[4];
  void* gather_output_mlu2 = gather_mlu(outputMluPtrS[7], input_shape, outputMluPtrS[5],index_shape,CNML_DIM_C,cnrt_queue);
  
  cnrtDestroyRuntimeContext(rt_ctx_);
  cnrtUnloadModel(model);
  cnrtDestroyFunction(function);

  gettimeofday(&tpend,NULL);
  execTime = 1000000 * (tpend.tv_sec - tpstart.tv_sec) +
    tpend.tv_usec - tpstart.tv_usec;
  LOG(INFO) << " cfnet1_gather time: " << execTime << " us";

  gettimeofday(&tpstart,NULL);

  cnrtModel_t model2;
  cnrtRuntimeContext_t ctx2;
  cnrtQueue_t queue2;
  std::string fname2 = (std::string)"cfnet2.cambricon";
  int size2;
  cnrtGetModelSize(fname2.c_str(), &size2);
  assert(cnrtLoadModel(&model2, fname2.c_str())==CNRT_RET_SUCCESS);
  cnrtFunction_t function2;
  std::string name2 = (std::string)"subnet0";
  assert(cnrtCreateFunction(&function2)==CNRT_RET_SUCCESS);
  assert(cnrtExtractFunction(&function2, model2, name2.c_str())==CNRT_RET_SUCCESS);
  assert(cnrtCreateRuntimeContext(&ctx2, function2, NULL)==CNRT_RET_SUCCESS);
  assert(cnrtSetRuntimeContextDeviceId(ctx2, 0)==CNRT_RET_SUCCESS);
  cnrtRet_t rett = cnrtInitRuntimeContext(ctx2, NULL);
  if (rett != CNRT_RET_SUCCESS){
    LOG(FATAL)<<"Failed to initialize runtime context";
  }

  cnrtRuntimeContextCreateQueue(ctx2,&queue2);

  int inputNum2;
  int outputNum2;
  int64_t* inputSizeS2 = nullptr;
  int64_t* outputSizeS2 = nullptr;
  
  cnrtGetInputDataSize(&inputSizeS2, &inputNum2, function2);
  cnrtGetOutputDataSize(&outputSizeS2, &outputNum2, function2);
  cnrtDataType_t* input_data_type2 = nullptr;
  cnrtDataType_t * output_data_type2 = nullptr;
  cnrtGetInputDataType(&input_data_type2, &inputNum2, function2);
  cnrtGetOutputDataType(&output_data_type2, &outputNum2, function2);

  void ** param2 = reinterpret_cast<void**>(malloc(sizeof(void*) * (inputNum2 + outputNum2)));
  void ** inputMluPtrS2 = reinterpret_cast<void**>(malloc(sizeof(void*) * inputNum2));
  void ** outputMluPtrS2 = reinterpret_cast<void**>(malloc(sizeof(void*) * outputNum2));


  cnrtFree(outputMluPtrS[3]);
  cnrtFree(outputMluPtrS[5]);
  cnrtFree(outputMluPtrS[7]);
  cnrtFree(outputMluPtrS[1]);

  for(int i = 0 ; i< inputNum2 ; i++){
    switch(i){
      case 0:
        inputMluPtrS2[14] = gather_output_mlu1;
      break;
      case 1:
        inputMluPtrS2[13] = outputMluPtrS[0];
      break;
      case 2:
        inputMluPtrS2[10] = outputMluPtrS[2];
      break;
      case 3:
        inputMluPtrS2[12] = gather_output_mlu2;
      break;
      case 4:
        inputMluPtrS2[11] = outputMluPtrS[4];
      break;
      case 5:
        inputMluPtrS2[9] = outputMluPtrS[6];
        break;
      case 6:
        inputMluPtrS2[8] =outputMluPtrS[8] ;
        break;
      case 7:
        inputMluPtrS2[7] = outputMluPtrS[9] ;
        break;
      case 8:
        inputMluPtrS2[5] = outputMluPtrS[10] ;
        break;
      case 9:
        inputMluPtrS2[6] = outputMluPtrS[11];
        break;
      case 10:
        inputMluPtrS2[4] = outputMluPtrS[12];
        break;
      case 11:
        inputMluPtrS2[3] = outputMluPtrS[13];
        break;
      case 12:
        inputMluPtrS2[1] = outputMluPtrS[14];
        break;
      case 13:
        inputMluPtrS2[0] = outputMluPtrS[15];
        break;
      case 14: 
      // 1x1x1x512
        int lin_ip = inputSizeS2[2] / cnrtDataTypeSize(input_data_type2[2]);
        assert(lin_ip == 512);
        auto databuf = reinterpret_cast<float*>(malloc(sizeof(float) * lin_ip));
        linspace(0.0, 511, 512, databuf);
        void * templin = reinterpret_cast<void*>(databuf);
        void * tempshape = reinterpret_cast<void*> (malloc(sizeof(float)*lin_ip));
        std::vector<int> sshape(4,1);
        int tempdimNum = 4;
        cnrtGetInputDataShape((int**)&sshape,&tempdimNum,2,function2);
        int tempdim_order[4] = {0, 2, 3, 1};
        int tempdim_shape[4] = {sshape[0], sshape[3],
                        sshape[1], sshape[2]};  // NCHW
        cnrtTransDataOrder(templin, CNRT_FLOAT32, tempshape,4,tempdim_shape, tempdim_order);
        free(templin);
        cnrtMalloc(&inputMluPtrS2[2], inputSizeS2[2]);
        void * temp_lin_cpu_data = (void*)malloc(inputSizeS2[2]);
        if(input_data_type2[2] != CNRT_FLOAT32){
          cnrtCastDataType(tempshape,
                        CNRT_FLOAT32,
                        temp_lin_cpu_data,
                        input_data_type2[2],
                        lin_ip,
                        nullptr);
        }
        else{
                          temp_lin_cpu_data = tempshape;
                        }
        free(tempshape);
        cnrtMemcpy(inputMluPtrS2[2],temp_lin_cpu_data,inputSizeS2[2],CNRT_MEM_TRANS_DIR_HOST2DEV);
        break;
    }
  }

  for(int i = 0; i < outputNum2; i++){
    assert(cnrtMalloc(&outputMluPtrS2[i], outputSizeS2[i])==CNRT_RET_SUCCESS);
  }
  for (int i = 0; i < inputNum2; i++) {
    param2[i] = inputMluPtrS2[i];
  }
  for (int i = 0; i < outputNum2; i++) {
    param2[inputNum2 + i] = outputMluPtrS2[i];
  }

  float event_time_use2;
  CNRT_CHECK(cnrtInvokeRuntimeContext(ctx2, param2, queue2, nullptr));
  if (cnrtSyncQueue(queue2) == CNRT_RET_SUCCESS) {
  } else {
    LOG(INFO) << " SyncQueue Error ";
  }

  for (int i = 0 ; i< inputNum2 ; i++){
    cnrtFree(inputMluPtrS2[i]);
  }


  gettimeofday(&tpend,NULL);
  execTime = 1000000 * (tpend.tv_sec - tpstart.tv_sec) +
    tpend.tv_usec - tpstart.tv_usec;
  LOG(INFO) << " cfnet2 time: " << execTime << " us";

  gettimeofday(&tpstart,NULL);

  std::vector<int> shape_input2(5,1); int dimNum2 = 5;
  cnrtGetOutputDataShape((int **)&shape_input2, &dimNum2, 3, function2);
  // LOG(INFO)<<"the shape of right_feature_map: "<<" "<<shape_input[1]<<" "<<shape_input[2]<<" "<<shape_input[3]<<" "<<shape_input[4];
  int input_shape2[] = {shape_input2[1], shape_input2[2], shape_input2[3], shape_input2[4]};
  std::vector<int> shape_index2(5,1); int dimNum_index2 = 5;
  cnrtGetOutputDataShape((int **)&shape_index2, &dimNum_index2, 1, function2);
  int index_shape2[] = {shape_index2[1], shape_index2[2], shape_index2[3], shape_index2[4]};
  void* gather_output1_mlu_cfnet2 = gather_mlu(outputMluPtrS2[3], input_shape2, outputMluPtrS2[1],index_shape2,CNML_DIM_C,queue2);
  
  cnrtGetOutputDataShape((int **)&shape_input2, &dimNum2, 7, function2);
  input_shape2[0] = shape_input2[1]; input_shape2[1] = shape_input2[2]; input_shape2[2] = shape_input2[3]; input_shape2[3] = shape_input2[4];
  cnrtGetOutputDataShape((int **)&shape_index2, &dimNum_index2, 5, function2);
  index_shape2[0] = shape_index2[1]; index_shape2[1] = shape_index2[2]; index_shape2[2] = shape_index2[3]; index_shape2[3] = shape_index2[4];
  void* gather_output2_mlu_cfnet2 = gather_mlu(outputMluPtrS2[7], input_shape2, outputMluPtrS2[5],index_shape2,CNML_DIM_C,queue2);
  
  cnrtDestroyRuntimeContext(ctx2);
  cnrtUnloadModel(model2);
  cnrtDestroyFunction(function2);
  cnrtDestroyQueue(cnrt_queue);
  cnrtDestroyQueue(queue2);

  cnrtFree(outputMluPtrS2[3]);
  cnrtFree(outputMluPtrS2[5]);
  cnrtFree(outputMluPtrS2[7]);
  cnrtFree(outputMluPtrS2[1]);

  gettimeofday(&tpend,NULL);
  execTime = 1000000 * (tpend.tv_sec - tpstart.tv_sec) +
    tpend.tv_usec - tpstart.tv_usec;
  LOG(INFO) << " cfnet2_gather time: " << execTime << " us";

  gettimeofday(&tpstart,NULL);

  cnrtModel_t model3;
  cnrtRuntimeContext_t ctx3;
  cnrtQueue_t queue3 ;
  std::string fname3= (std::string)"cfnet3.cambricon";
  // LOG(INFO)<<"load file: "<<fname3;
  int size3;
  cnrtGetModelSize(fname3.c_str(),&size3);
  // LOG(INFO)<<"model size"<<size3;
  assert(cnrtLoadModel(&model3,fname3.c_str())==CNRT_RET_SUCCESS);
  cnrtFunction_t function3;
  std::string name3 = (std::string)"subnet0";
  assert(cnrtCreateFunction(&function3)==CNRT_RET_SUCCESS);
  assert(cnrtExtractFunction(&function3, model3, name3.c_str())==CNRT_RET_SUCCESS);
  assert(cnrtCreateRuntimeContext(&ctx3, function3, NULL)==CNRT_RET_SUCCESS);
  assert(cnrtSetRuntimeContextDeviceId(ctx3, 0)==CNRT_RET_SUCCESS);
  cnrtRet_t initret = cnrtInitRuntimeContext(ctx3, NULL);
  if (initret != CNRT_RET_SUCCESS){
    LOG(FATAL)<<"Failed to initialize runtime context";
  }

  cnrtRuntimeContextCreateQueue(ctx3,&queue3);

  int inputNum3;
  int outputNum3;
  int64_t* inputSizeS3= nullptr;
  int64_t* outputSizeS3 = nullptr;
  cnrtGetInputDataSize(&inputSizeS3, &inputNum3, function3);
  cnrtGetOutputDataSize(&outputSizeS3, &outputNum3, function3);
  cnrtDataType_t* input_data_type3 = nullptr;
  cnrtDataType_t * output_data_type3 = nullptr;
  cnrtGetInputDataType(&input_data_type3, &inputNum3, function3);
  cnrtGetOutputDataType(&output_data_type3, &outputNum3, function3);
  void ** param3 = reinterpret_cast<void**>(malloc(sizeof(void*) * (inputNum3 + outputNum3)));
  void ** inputMluPtrS3 = reinterpret_cast<void**>(malloc(sizeof(void*) * inputNum3));
  void ** outputMluPtrS3 = reinterpret_cast<void**>(malloc(sizeof(void*) * outputNum3));

  for(int i = 0 ; i<  inputNum3; i++){
    switch(i){
      case 0 :
        inputMluPtrS3[9] = gather_output1_mlu_cfnet2;
      break;
      case 1 :
        inputMluPtrS3[7] = gather_output2_mlu_cfnet2;
      break;  
      case 2 :
        inputMluPtrS3[8] = outputMluPtrS2[0]; 
      break;
      case 3 :
        inputMluPtrS3[5] = outputMluPtrS2[2];
      break;
      case 4 :
        inputMluPtrS3[6] = outputMluPtrS2[4];
      break;
      case 5 :
        inputMluPtrS3[4] = outputMluPtrS2[6];
      break;
      case 6 :
        inputMluPtrS3[3] = outputMluPtrS2[8];
      break;
      case 7 :
        inputMluPtrS3[2] = outputMluPtrS[16];
      break;
      case 8 :
        inputMluPtrS3[0] = outputMluPtrS[17];
      break;
      case 9 :
        inputMluPtrS3[1] = outputMluPtrS[18];
      break;
    }
  }

  for(int i = 0; i < outputNum3; i++){
    assert(cnrtMalloc(&outputMluPtrS3[i], outputSizeS3[i])==CNRT_RET_SUCCESS);
  }
  for (int i = 0; i < inputNum3; i++) {
    param3[i] = inputMluPtrS3[i];
  }
  for (int i = 0; i < outputNum3; i++) {
    param3[inputNum3 + i] = outputMluPtrS3[i];
  }

  float event_time_use3;
  // LOG(INFO)<<"start cfnet3";
  CNRT_CHECK(cnrtInvokeRuntimeContext(ctx3, param3, queue3, nullptr));
  // LOG(INFO)<<"end cfnet3";
  if (cnrtSyncQueue(queue3) == CNRT_RET_SUCCESS) {
    // LOG(INFO) << "cfnet3 Done!" ;
  } else {
    LOG(INFO) << " SyncQueue Error ";
  }

  cnrtDestroyRuntimeContext(ctx3);
  cnrtUnloadModel(model3);
  cnrtDestroyFunction(function3);
  cnrtDestroyQueue(queue3);

  for(int i = 0 ; i< inputNum3 ; i++){
    assert(cnrtFree(inputMluPtrS3[i])==CNRT_RET_SUCCESS);
  }
  for(int i = 0; i < outputNum3; i++){
    assert(cnrtFree(outputMluPtrS3[i])==CNRT_RET_SUCCESS);
  }

  gettimeofday(&tpend, NULL);
  execTime = 1000000 * (tpend.tv_sec - tpstart.tv_sec) +
    tpend.tv_usec - tpstart.tv_usec;
  LOG(INFO) << " cfnet3 time: " << execTime << " us";
  cnrtDestroy();
  return 0;
}
#else
int main() {
  LOG(FATAL) << "This program should be compiled with USE_MLU!";
  return 0;
}
#endif  // USE_MLU
