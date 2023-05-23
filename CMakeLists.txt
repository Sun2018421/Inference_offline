# Collect source files
file(GLOB_RECURSE srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
find_package(glog 0.6.0 REQUIRED)
# Build each source file independently
foreach(source ${srcs})
    # get file name
    get_filename_component(name ${source} NAME_WE)

    # get folder name
    get_filename_component(path ${source} PATH)
    get_filename_component(folder ${path} NAME_WE)

    # target
    add_executable(${name} ${source})
    set(NEUWARE_HOME "/usr/local/neuware")
    add_definitions(-DUSE_MLU)
    include_directories(${CNRT_INCLUDE_DIRS} ${INCLUDE_DIRS})
    message("NEUWARE_HOME is ${NEUWARE_HOME}")
    include_directories(${NEUWARE_HOME}/include)
    link_directories(${NEUWARE_HOME}/lib64)
    target_link_libraries(${name} glog::glog)
    target_link_libraries(${name} ${LINKER_LIBS} ${CNRT_LIBS})
    target_link_libraries(${name} ${NEUWARE_HOME}/lib64/libcnrt.so)
    target_link_libraries(${name} ${NEUWARE_HOME}/lib64/libcnml.so)
endforeach()
