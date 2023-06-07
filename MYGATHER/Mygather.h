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

void * gather_mlu(void * & input_tensor, int * input_shape, void * &index_tensor, int * index_shape , cnmlDimension_t dim, cnrtQueue_t &cnrt_queue);