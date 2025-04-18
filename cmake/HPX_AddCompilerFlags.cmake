# C++23 support
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(HPX_CXX23_FLAG "-std=c++23")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(HPX_CXX23_FLAG "-std=c++23")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(HPX_CXX23_FLAG "/std:c++23")
endif() 