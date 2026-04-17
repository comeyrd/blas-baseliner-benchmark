set(sources main.cpp)
add_executable(baseliner-blas ${sources})
if(HAS_CUDA)
    add_subdirectory(cuda)
elseif(HAS_HIP)
    add_subdirectory(hip)
endif()
target_link_libraries(baseliner-blas PUBLIC baseliner::baseliner ${BLAS_LIBRARIES})
target_include_directories(baseliner-blas PRIVATE 
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
)