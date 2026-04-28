add_library(blas_sources INTERFACE)
target_sources(blas_sources INTERFACE main.cpp)

if(HAS_CUDA)
    add_subdirectory(cuda)
elseif(HAS_HIP)
    add_subdirectory(hip)
endif()

 add_executable(baseliner-blas ${BLAS_SOURCES})
 target_link_libraries(baseliner-blas PUBLIC baseliner_adapters::baseliner ${BLAS_LIBRARIES})
 target_link_libraries(baseliner-blas PRIVATE blas_sources)
 target_include_directories(baseliner-blas PRIVATE 
     "${CMAKE_CURRENT_SOURCE_DIR}/.."
 )

add_executable(nvbench-blas)
target_link_libraries(nvbench-blas PUBLIC baseliner_adapters::nvbench nvbench::main ${BLAS_LIBRARIES})
target_link_libraries(nvbench-blas PRIVATE blas_sources)
target_include_directories(nvbench-blas PRIVATE 
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
)