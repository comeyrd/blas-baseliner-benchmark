target_sources(baseliner-blas PRIVATE
   RocBlasHelper.hip)
add_subdirectory(Gemm)