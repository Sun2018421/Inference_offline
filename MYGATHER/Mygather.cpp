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
#include "Mygather.h"
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