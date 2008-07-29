cmake_minimum_required(VERSION 2.6)
project (mstack CXX C)

# We need to use Boost, namely the following Boost libraries
set (Boost_COMPONENTS_NEEDED
    date_time 
    filesystem 
    graph 
    regex 
    serialization 
    thread 
    system 
    signals
)
find_package(Boost 1.35.0 COMPONENTS ${Boost_COMPONENTS_NEEDED})

INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
LINK_DIRECTORIES(${Boost_LIBRARY_DIR})

set (mstack_SOURCES /home/mark/mstack_hpx/mstack.cpp)
set (hpx_LIBRARIES /home/mark/hpx/src/libhpx.a)
add_executable (mstack ${mstack_SOURCES})
target_link_libraries(mstack ${hpx_LIBRARIES} ${Boost_LIBRARIES})
