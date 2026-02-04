cmake_minimum_required(VERSION 3.17)
project(task_bench VERSION 1.0.0 LANGUAGES C CXX)

find_package(MPI)

set(SOURCES
    hpx_task_bench.cpp
    core.cc
    core.h
    core_c.cc
    core_c.h
    core_kernel.cc
    core_kernel.h
    timer.cc
    timer.h
    core_random.c
    core_random.h
    siphash.c
    siphash.h
)

if(COMMAND add_hpx_executable)
    add_hpx_executable(
        task_bench
        SOURCES ${SOURCES}
        FOLDER "Examples/TaskBench"
    )
else()
    find_package(HPX REQUIRED)
    add_executable(task_bench ${SOURCES})
    target_link_libraries(task_bench PUBLIC HPX::hpx)
endif()

if(MPI_FOUND)
    target_link_libraries(task_bench PUBLIC MPI::MPI_CXX)
    target_compile_definitions(task_bench PRIVATE HAVE_MPI)
else()
    target_compile_definitions(task_bench PRIVATE NO_MPI)
endif()
