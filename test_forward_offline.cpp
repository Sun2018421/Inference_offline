#include <sys/time.h>
#include "glog/logging.h"
#ifdef USE_MLU
#include <cnrt.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

DEFINE_int32(mludevice, 0, "set using mlu device number, default: 0");
DEFINE_int32(input_data_count, -1, "the input file count,default: -1, get rand  data");
// support three input data
DEFINE_string(input1, " ", "the input file name of the offline model");
DEFINE_string(input2, " ", "the input file name of the offline model");
DEFINE_string(input3, " ", "the input file name of the offline model");
void rand1(float* data, int length) {
  unsigned int seed = 1024;
  for (int i = 0; i < length; ++i) {
    if (i % 5 == 4) {
      data[i] = rand_r(&seed) % 100 / 100. + 0.0625;
    } else if (i % 5 >= 2) {
      data[i] = data[i - 2] + (rand_r(&seed) % 100) / 100.0 + 0.0625;
    } else {
      data[i] = (rand_r(&seed) % 100) / 100. + 0.0625;
    }
  }
}

void rand2(float* data, int length) {
  unsigned int seed = 1024;
  for (int i = 0; i < length; ++i) {
    if (i % 5 == 0) {
      data[i] = rand_r(&seed) % 100 / 100. + 0.0625;
    } else if (i % 5 > 2) {
      data[i] = data[i - 2] + (rand_r(&seed) % 100) / 100.0 + 0.0625;
    } else {
      data[i] = (rand_r(&seed) % 100) / 100. + 0.0625;
    }
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

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc < 4) {
    LOG(INFO) << "USAGE: " << argv[0] << ": <cambricon_file>"
              << " <output_file> <function_name0> <function_name1> ...";
    return 1;
  }
  cnrtInit(0);
  unsigned devNum;
  cnrtGetDeviceCount(&devNum);
  if (FLAGS_mludevice >= 0) {
    CHECK_NE(devNum, 0) << "No device found";
    CHECK_LE(FLAGS_mludevice, devNum) << "valid device count: " << devNum;
  } else {
    LOG(FATAL) << "Invalid device number";
  }

  cnrtDev_t dev;
  cnrtGetDeviceHandle(&dev, FLAGS_mludevice);
  cnrtSetCurrentDevice(dev);
  // 2. load model and get function
  cnrtModel_t model;
  std::string fname = (std::string)argv[1];
  LOG(INFO) << "load file: " << fname;
  int size;
  cnrtGetModelSize(fname.c_str(), &size);
  LOG(INFO) << "model size: " << size;
  cnrtLoadModel(&model, fname.c_str());
  cnrtFunction_t function;
  cnrtRuntimeContext_t rt_ctx_;
  unsigned int in_n, in_c, in_h, in_w;
  unsigned int out_n, out_c, out_h, out_w;

  struct timeval tpend, tpstart;
  gettimeofday(&tpstart, NULL);
  for (int n = 3; n < argc; n++) {
    std::string name = (std::string)argv[n];
    cnrtCreateFunction(&function);
    cnrtExtractFunction(&function, model, name.c_str());
    cnrtCreateRuntimeContext(&rt_ctx_, function, NULL);
    // 3. get function's I/O DataDesc
    int inputNum, outputNum;
    int64_t* inputSizeS = nullptr;
    int64_t* outputSizeS = nullptr;
    cnrtGetInputDataSize(&inputSizeS, &inputNum, function);
    cnrtGetOutputDataSize(&outputSizeS, &outputNum, function);
    cnrtDataType_t* input_data_type = nullptr;
    cnrtDataType_t* output_data_type = nullptr;
    cnrtGetInputDataType(&input_data_type, &inputNum, function);
    cnrtGetOutputDataType(&output_data_type, &outputNum, function);
    // 4. allocate I/O data space on CPU memory and prepare Input data
    void** inputCpuPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * inputNum));
    void** outputCpuPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * outputNum));
    std::vector<float*> output_cpu;
    std::vector<int> in_count;
    std::vector<int> out_count;
    void** param =
        reinterpret_cast<void**>(malloc(sizeof(void*) * (inputNum + outputNum)));
    srand(10);
    std::vector<std::string> input_file_name;
    input_file_name.push_back(FLAGS_input1);
    input_file_name.push_back(FLAGS_input2);
    input_file_name.push_back(FLAGS_input3);
    for (int i = 0; i < inputNum; i++) {
      int ip = inputSizeS[i] / cnrtDataTypeSize(input_data_type[i]);
      auto databuf = reinterpret_cast<float*>(malloc(sizeof(float) * ip));
      if (FLAGS_input_data_count != -1) {
        CHECK_EQ(inputNum, FLAGS_input_data_count) << "check yout input count";
        CHECK_LT(FLAGS_input_data_count, 3) << "only support input_data_count <=3";
        readData(databuf, ip, input_file_name[FLAGS_input_data_count-1-i]);
      } else {
         if (i == 0) {
           rand1(databuf, ip);
         } else if (i == 1) {
           rand2(databuf, ip);
         }
      }

      in_count.push_back(ip);
      inputCpuPtrS[i] = reinterpret_cast<void*>(databuf);  // NCHW
      std::vector<int> shape(4, 1);
      int dimNum = 4;
      cnrtGetInputDataShape((int**)&shape, &dimNum, i, function);  // NHWC
      in_n = shape[0];
      in_c = (input_data_type[i] == CNRT_UINT8) ? (shape[3] - 1) : shape[3];
      in_h = shape[1];
      in_w = shape[2];
      LOG(INFO) << "in_n " << in_n << " in_c " << in_c
                << " in_h " << in_h << " in_w " << in_w;
    }
    for (int i = 0; i < outputNum; i++) {
      int op = outputSizeS[i] / cnrtDataTypeSize(output_data_type[i]);
      float* outcpu = reinterpret_cast<float*>(malloc(op * sizeof(float)));
      out_count.push_back(op);
      output_cpu.push_back(outcpu);
      outputCpuPtrS[i] = reinterpret_cast<void*>(outcpu);
      std::vector<int> shape(4, 1);
      int dimNum = 4;
      cnrtGetOutputDataShape((int**)&shape, &dimNum, i, function);  // NHWC
      out_n = shape[0];
      out_c = shape[3];
      out_h = shape[1];
      out_w = shape[2];
      LOG(INFO) << "out_n " << out_n << " out_c " << out_c
                << " out_h " << out_h << " out_w " << out_w;
    }
    // 5. allocate I/O data space on MLU memory and copy Input data
    // Only 1 batch so far
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
    // 6. create cnrt_queue
    cnrtQueue_t cnrt_queue;
    cnrtCreateQueue(&cnrt_queue);

    cnrtSetRuntimeContextDeviceId(rt_ctx_, dev);
    cnrtInitRuntimeContext(rt_ctx_, NULL);
    void** tempPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * inputNum));
    void* temp_input_cpu_data = nullptr;
    for (int i = 0; i < inputNum; i++) {
      int ip = inputSizeS[i] / cnrtDataTypeSize(input_data_type[i]);
      auto databuf = reinterpret_cast<float*>(malloc(sizeof(float) * ip));
      tempPtrS[i] = reinterpret_cast<void*>(databuf);
      std::vector<int> shape(4, 1);
      int dimNum = 4;
      cnrtGetInputDataShape((int**)&shape, &dimNum, i, function);
      int dim_order[4] = {0, 2, 3, 1};
      int dim_shape[4] = {shape[0], shape[3],
                          shape[1], shape[2]};  // NCHW
      cnrtTransDataOrder(inputCpuPtrS[i], CNRT_FLOAT32, tempPtrS[i],
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
    // create start_event and end_event
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
      LOG(INFO) << " hardware time: " << event_time_use;
    } else {
      LOG(INFO) << " SyncQueue Error ";
    }
    void** outPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * outputNum));
    void* temp_output_cpu_data = nullptr;
    for (int i = 0; i < outputNum; i++) {
      temp_output_cpu_data = (void*)malloc(outputSizeS[i]);
      cnrtMemcpy(temp_output_cpu_data,
                 outputMluPtrS[i],
                 outputSizeS[i],
                 CNRT_MEM_TRANS_DIR_DEV2HOST);
      int output_count = outputSizeS[i] / cnrtDataTypeSize(output_data_type[i]);
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
      auto databuf = reinterpret_cast<float*>(malloc(sizeof(float) * output_count));
      outPtrS[i] = reinterpret_cast<void*>(databuf);
      std::vector<int> shape(4, 1);
      int dimNum = 4;
      cnrtGetOutputDataShape((int**)&shape, &dimNum, i, function);
      int dim_order[4] = {0, 3, 1, 2};
      int dim_shape[4] = {shape[0], shape[1],
                          shape[2], shape[3]};  // NHWC
      cnrtTransDataOrder(outputCpuPtrS[i], CNRT_FLOAT32, outPtrS[i],
                         4, dim_shape, dim_order);

      if (temp_output_cpu_data) {
        free(temp_output_cpu_data);
        temp_output_cpu_data = nullptr;
      }
    }
    for (int i = 0; i < outputNum; i++) {
      LOG(INFO) << "copying output data of " << i << "th" << " function: " << argv[n];
      std::stringstream ss;
      if (outputNum > 1) {
        ss << argv[2] << "_" << argv[n] << i;
      } else {
        ss << argv[2] << "_" << argv[n];
      }
      std::string output_name = ss.str();
      LOG(INFO) << "writing output file of segment " << argv[n] << " output: "
                << i << "th" << " output file name: " << output_name;
      std::ofstream fout(output_name, std::ios::out);
      fout << std::flush;
      for (int j = 0; j < out_count[i]; ++j) {
        fout <<(reinterpret_cast<float*>(outPtrS[i]))[j] << std::endl;
      }
      fout << std::flush;
      fout.close();
    }
    for (auto flo : output_cpu) {
      free(flo);
    }
    output_cpu.clear();
    // 8. free memory space
    free(inputCpuPtrS);
    free(outputCpuPtrS);
    free(outPtrS);
    for (int i = 0; i < inputNum; i++)
      cnrtFree(inputMluPtrS[i]);
    for (int i = 0; i < outputNum; i++)
      cnrtFree(outputMluPtrS[i]);
    cnrtDestroyQueue(cnrt_queue);
    cnrtDestroyFunction(function);
  }
  cnrtUnloadModel(model);
  gettimeofday(&tpend, NULL);
  float execTime = 1000000 * (tpend.tv_sec - tpstart.tv_sec) +
    tpend.tv_usec - tpstart.tv_usec;
  LOG(INFO) << " execution time: " << execTime << " us";
  cnrtDestroy();
  return 0;
}
#else
int main() {
  LOG(FATAL) << "This program should be compiled with USE_MLU!";
  return 0;
}
#endif  // USE_MLU
