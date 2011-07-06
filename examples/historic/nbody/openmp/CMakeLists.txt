cmake_minimum_required(VERSION 2.6)
PROJECT(ompbh)

ADD_EXECUTABLE (ompbh ompbh.cpp)

# Find OPENMP flag
find_package( OpenMP )
if ( OPENMP_FOUND )
set_property(
     TARGET  ompbh
     APPEND
     PROPERTY COMPILE_FLAGS ${OpenMP_CXX_FLAGS}
  )
set_property(
     TARGET ompbh
     APPEND
     PROPERTY LINK_FLAGS ${OpenMP_CXX_FLAGS}
  )
else ( OPENMP_FOUND )
  message(STATUS "Warning: Parallelization disabled. Compiler doesn't support OpenMP")
endif ( OPENMP_FOUND )
