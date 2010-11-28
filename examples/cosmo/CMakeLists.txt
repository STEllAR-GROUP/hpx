# Copyright (c) 2007-2010 Hartmut Kaiser
# Copyright (c) 2009-2010 Matt Anderson
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# just recurse into all components subdirectories and execute the scripts there
set(cosmo_amr_SUDIRS 
    amr amr_c amr_c_test)

find_package(GMP)
find_package(SDF)

if(SDF_FOUND)
    add_definitions(-DSDF_FOUND)
    include_directories(${SDF_INCLUDE_DIR})
endif()

if(GMP_FOUND)
    add_definitions(-DGMP_FOUND)
    include_directories(${GMP_INCLUDE_DIR})
    find_package(MPFR)
    if(MPFR_FOUND)
        add_definitions(-DMPFR_FOUND=1)
        add_definitions(-DMPFR_USE_NED_ALLOCATOR=1)
        include_directories(${MPFR_INCLUDE_DIR})

        # allow the mpreal class to take advantage of move semantics
        add_definitions(-DMPFR_USE_BOOST_MOVE=1)
        include_directories(${hpx_SOURCE_DIR}/external/move)
    endif()
endif()

foreach(subdir ${cosmo_amr_SUDIRS})
    add_subdirectory(${subdir})
endforeach()

# define build target for this directory
set(cosmo_amr_client_SOURCES 
    amr_client.cpp)

if (MPFR_FOUND) 
    set(cosmo_amr_client_SOURCES 
        ${cosmo_amr_client_SOURCES}
        mpreal.cpp 
        serialize_mpreal.cpp)
endif()

set(cosmo_amr_DEPENDENCIES
    cosmo_amr_component 
    cosmo_amr_c_component 
    cosmo_amr_c_test_component
    distributing_factory_component)

if(GMP_FOUND)
    set(cosmo_amr_DEPENDENCIES 
        ${cosmo_amr_DEPENDENCIES}
        ${GMP_LIBRARY})
    if(MPFR_FOUND)
        set(cosmo_amr_DEPENDENCIES 
            ${cosmo_amr_DEPENDENCIES}
            ${MPFR_LIBRARY})
    endif()
endif()

if(SDF_FOUND)
    set(cosmo_amr_DEPENDENCIES 
        ${cosmo_amr_DEPENDENCIES}
        ${SDF_LIBRARY})
endif()

add_hpx_executable(cosmo_amr_client 
    SOURCES ${cosmo_amr_client_SOURCES}
    DEPENDENCIES ${cosmo_amr_DEPENDENCIES})

