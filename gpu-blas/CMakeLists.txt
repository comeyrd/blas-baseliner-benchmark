add_library(blas_sources INTERFACE)
target_sources(blas_sources INTERFACE main.cpp)

if(HAS_CUDA)
    add_subdirectory(cuda)
elseif(HAS_HIP)
    add_subdirectory(hip)
endif()
option(BASELINER_BLAS_BUILD_BASELINER "Build baseliner-blas executable" ON)
option(BASELINER_BLAS_BUILD_MINIMAL   "Build minimal-blas executable"   OFF)
option(BASELINER_BLAS_BUILD_NVBENCH   "Build nvbench-blas executable"   OFF)
option(BASELINER_BLAS_BUILD_PRIMBENCH "Build primbench-blas executable" OFF)

if(BASELINER_BLAS_BUILD_BASELINER)
    add_executable(baseliner-blas)
    target_link_libraries(baseliner-blas PUBLIC baseliner_adapters::baseliner ${BLAS_LIBRARIES})
    target_link_libraries(baseliner-blas PRIVATE blas_sources)
    target_include_directories(baseliner-blas PRIVATE 
        "${CMAKE_CURRENT_SOURCE_DIR}/.."
    )
endif()
if(BASELINER_BLAS_BUILD_MINIMAL)
    add_executable(minimal-blas)
    target_link_libraries(minimal-blas PUBLIC baseliner_adapters::minimal ${BLAS_LIBRARIES})
    target_link_libraries(minimal-blas PRIVATE blas_sources)
    target_include_directories(minimal-blas PRIVATE 
        "${CMAKE_CURRENT_SOURCE_DIR}/.."
    )
endif()

if(BASELINER_BLAS_BUILD_NVBENCH AND HAS_CUDA)
    add_executable(nvbench-blas)
    target_link_libraries(nvbench-blas PUBLIC baseliner_adapters::nvbench nvbench::main ${BLAS_LIBRARIES})
    target_link_libraries(nvbench-blas PRIVATE blas_sources)
    target_include_directories(nvbench-blas PRIVATE 
        "${CMAKE_CURRENT_SOURCE_DIR}/..")
endif()
if(BASELINER_BLAS_BUILD_PRIMBENCH)
    add_executable(primbench-blas)
    target_link_libraries(primbench-blas PUBLIC baseliner_adapters::primbench ${BLAS_LIBRARIES})
    target_link_libraries(primbench-blas PRIVATE blas_sources)
    target_include_directories(primbench-blas PRIVATE 
    "${CMAKE_CURRENT_SOURCE_DIR}/..")
endif()
