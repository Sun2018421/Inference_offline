#include <cnrt.h>
#include <iostream>
#include <stdio.h>
#include <cnml.h>

int main(){
    CNRT_CHECK(cnrtInit(0));
    cnrtDev_t dev;
    CNRT_CHECK(cnrtGetDeviceHandle(&dev, 0));
    CNRT_CHECK(cnrtSetCurrentDevice(dev));

    const cnmlCoreVersion_t core_version = CNML_MLU270;
    const int coreNum = 1;
    const int dimNum=4;
    const int n = 8, c = 2, h = 2, w = 2;
    const int indexn = 4, indexc = 1, indexh = 1, indexw = 1;
    int input_count_1 = n * h  * w * c;
    int input_count_2 = indexn * indexc  * indexh * indexw;
    int output_count = indexn * h  * w * c;

    float * input_cpu_ptr_1 = (float*) malloc(input_count_1 * sizeof(float));
    int * input_cpu_ptr_2 = (int*) malloc(input_count_2 * sizeof(int)); // index_tensor
    float * output_cpu_ptr = (float*) malloc(output_count * sizeof(float));

    unsigned int seed = time(0);
    for(int i = 0 ; i < input_count_1 ; i ++) input_cpu_ptr_1[i] = i ;
    for(int i = 0 ; i < input_count_2 ; i ++) input_cpu_ptr_2[i] = input_count_2 - (i+1) ;

    int input_shape_1 [] = {n, c, h, w};
    int input_shape_2 [] = {indexn, indexc, indexh, indexw};
    int output_shape [] = {indexn, c, h, w};

    cnmlTensor_t input_tensor_1 = NULL;
    cnmlCreateTensor_V2(&input_tensor_1, CNML_TENSOR);
    cnmlSetTensorShape_V2(input_tensor_1, dimNum, input_shape_1, NULL);
    cnmlSetTensorDataType(input_tensor_1, CNML_DATA_FLOAT32);

    cnmlTensor_t input_tensor_2 = NULL;
    cnmlCreateTensor_V2(&input_tensor_2, CNML_TENSOR);
    cnmlSetTensorShape_V2(input_tensor_2, dimNum, input_shape_2, NULL);
    cnmlSetTensorDataType(input_tensor_2, CNML_DATA_INT32);

    cnmlTensor_t output_tensor = NULL;
    cnmlCreateTensor_V2(&output_tensor, CNML_TENSOR);
    cnmlSetTensorShape_V2(output_tensor, dimNum, output_shape, NULL);
    cnmlSetTensorDataType(output_tensor, CNML_DATA_FLOAT32);

    cnmlDimension_t dim = CNML_DIM_N;
    cnmlBaseOp_t gather_op = NULL;
    cnmlCreateGatherV2Op(&gather_op, input_tensor_1, input_tensor_2, output_tensor, dim);

    cnmlSetBaseOpCoreVersion(gather_op, core_version);
    cnmlSetBaseOpCorenum(gather_op, coreNum);
    cnmlCompileBaseOp_V2(gather_op);

    void * input_mlu_ptr_1 = NULL;
    void * input_mlu_ptr_2 = NULL;
    void * output_mlu_ptr = NULL;

    cnrtMalloc(&input_mlu_ptr_1, input_count_1 * sizeof(float));
    cnrtMalloc(&input_mlu_ptr_2, input_count_2 * sizeof(int));
    cnrtMalloc(&output_mlu_ptr, output_count * sizeof(float));

    cnrtMemcpy(input_mlu_ptr_1, input_cpu_ptr_1, input_count_1 * sizeof(float), CNRT_MEM_TRANS_DIR_HOST2DEV);
    cnrtMemcpy(input_mlu_ptr_2, input_cpu_ptr_2, input_count_2* sizeof(int), CNRT_MEM_TRANS_DIR_HOST2DEV);

    cnrtQueue_t queue;
    cnrtCreateQueue(&queue);

    cnmlComputeGatherV2OpForward_V4(gather_op, NULL, input_mlu_ptr_1, NULL,  input_mlu_ptr_2, NULL, output_mlu_ptr, queue, NULL);

    cnrtSyncQueue(queue);
    cnrtDestroyQueue(queue);

    cnrtMemcpy(output_cpu_ptr, output_mlu_ptr, output_count* sizeof(float), CNRT_MEM_TRANS_DIR_DEV2HOST);

    cnmlDumpTensor2File_V2("MLU_GATHER_OUTPUT", output_tensor, output_cpu_ptr, false);

    cnmlDestroyBaseOp(&gather_op);

    cnrtFree(input_mlu_ptr_1);
    cnrtFree(input_mlu_ptr_2);
    cnrtFree(output_mlu_ptr);

    cnmlDestroyTensor(&input_tensor_1);
    cnmlDestroyTensor(&input_tensor_2);
    cnmlDestroyTensor(&output_tensor);

    free(input_cpu_ptr_1);
    free(input_cpu_ptr_2);
    free(output_cpu_ptr);

    return  0;
}
