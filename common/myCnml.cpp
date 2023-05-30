#include <cnml.h>
#include <cnrt.h>
#include <assert.h>
#include "include/myCnml.h"

void * mygather(void * & input_tensor, \
 int * input_shape, void * &index_tensor, int * index_shape, \
 cnmlDimension_t dim, cnrtQueue_t &cnrt_queue ){
    cnmlBaseOp_t cast_op = NULL;
    cnmlTensor_t cast_input = NULL;
    cnmlCreateTensor_V2(&cast_input, CNML_TENSOR);
    cnmlSetTensorShape_V2(cast_input, 4, index_shape, NULL);
 }