#include <cnml.h>
#include <cnrt.h>
#include <assert.h>

#ifndef MYGATHER_HPP_
#define MYGATHER_HPP_
void *mygather(void * & input_tensor, \
 int * input_shape, void * &index_tensor, int * index_shape, \
 cnmlDimension_t dim, cnrtQueue_t &cnrt_queue );

#endif