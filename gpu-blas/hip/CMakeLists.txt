target_sources(blas_sources INTERFACE
   RocBlasHelper.hip)
add_subdirectory(Gemm)