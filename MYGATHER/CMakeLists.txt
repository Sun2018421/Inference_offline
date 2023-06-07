
add_library(MYGATHER Mygather.cpp)
set(NEUWARE_HOME "/usr/local/neuware")
include_directories(${CNRT_INCLUDE_DIRS} ${INCLUDE_DIRS})
include_directories(${NEUWARE_HOME}/include)
include_directories()
link_directories(${NEUWARE_HOME}/lib64)

target_link_libraries(MYGATHER glog::glog)
target_link_libraries(MYGATHER ${LINKER_LIBS} ${CNRT_LIBS})
target_link_libraries(MYGATHER ${NEUWARE_HOME}/lib64/libcnrt.so)
target_link_libraries(MYGATHER ${NEUWARE_HOME}/lib64/libcnml.so)