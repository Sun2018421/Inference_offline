cmake_minimum_required(VERSION 3.21)
project(Three_offline_models)
    # target
find_package(glog 0.6.0 REQUIRED)
# add_executable(fnet_forward_offline fnet_forward_offline.cpp)
add_executable(test_gather test_gather.cpp)
set(NEUWARE_HOME "/usr/local/neuware")
add_definitions(-DUSE_MLU)
include_directories(${CNRT_INCLUDE_DIRS} ${INCLUDE_DIRS})

add_subdirectory(MYGATHER)
message("NEUWARE_HOME is ${NEUWARE_HOME}")
include_directories(${NEUWARE_HOME}/include)
link_directories(${NEUWARE_HOME}/lib64)
# 临时注释掉测试test_gather
# target_link_libraries(fnet_forward_offline glog::glog)
# target_link_libraries(fnet_forward_offline ${LINKER_LIBS} ${CNRT_LIBS})
# target_link_libraries(fnet_forward_offline ${NEUWARE_HOME}/lib64/libcnrt.so)
# target_link_libraries(fnet_forward_offline ${NEUWARE_HOME}/lib64/libcnml.so)

target_link_libraries(test_gather glog::glog)
target_link_libraries(test_gather ${LINKER_LIBS} ${CNRT_LIBS})
target_link_libraries(test_gather ${NEUWARE_HOME}/lib64/libcnrt.so)
target_link_libraries(test_gather ${NEUWARE_HOME}/lib64/libcnml.so)
target_link_libraries(test_gather MYGATHER)


