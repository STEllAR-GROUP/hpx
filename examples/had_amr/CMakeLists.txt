# Copyright (c) 2007-2010 Hartmut Kaiser
# Copyright (c) 2009-2010 Matt Anderson
# Copyright (c) 2011      Bryce Lelbach 
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# just recurse into all components subdirectories and execute the scripts there
set(hpx_HAD_AMR_SUBDIRS 
    amr
    amr_c
    amr_c_test)

find_package(GMP)
find_package(SDF)

if(SDF_FOUND)
  # FIXME: HPX_* identifiers please
  add_definitions(-DSDF_FOUND)
  include_directories(${SDF_INCLUDE_DIR})
endif()

if(GMP_FOUND)
  # FIXME: HPX_* identifiers please
  add_definitions(-DGMP_FOUND)
  include_directories(${GMP_INCLUDE_DIR})

  # MPFR depends on GMP, so no point looking for it if we don't have GMP  
  find_package(MPFR)
  
  if(MPFR_FOUND)
    # FIXME: HPX_* identifiers please
    add_definitions(-DMPFR_FOUND=1)
    add_definitions(-DMPFR_USE_NED_ALLOCATOR=1)
    
    include_directories(${MPFR_INCLUDE_DIR})

    # allow the mpreal class to take advantage of move semantics
    # FIXME: HPX_* identifiers please
    add_definitions(-DMPFR_USE_BOOST_MOVE=1)
    include_directories(${hpx_SOURCE_DIR}/external/move)
  endif()
endif()

foreach(hpx_HAD_AMR_SUBDIR ${hpx_HAD_AMR_SUBDIRS})
  add_subdirectory(${hpx_HAD_AMR_SUBDIR})
endforeach()

# define build target for this directory
set(had_amr_client_SOURCES 
    amr_client.cpp)

if (MPFR_FOUND) 
    set(had_amr_client_SOURCES 
        ${had_amr_client_SOURCES}
        mpreal.cpp 
        serialize_mpreal.cpp)
endif()

# define basic dependencies
set(had_amr_DEPENDENCIES
    had_amr_component 
    had_amr_c_component 
    had_amr_c_test_component
    distributing_factory_component)

if(GMP_FOUND)
  # add GMP as a dependency
  set(had_amr_DEPENDENCIES 
      ${had_amr_DEPENDENCIES}
      ${GMP_LIBRARY})
  if(MPFR_FOUND)
    # add MPFR as a dependency
    set(had_amr_DEPENDENCIES 
        ${had_amr_DEPENDENCIES}
        ${MPFR_LIBRARY})
  endif()
endif()

if(SDF_FOUND)
  # add SDF as a dependency
  set(had_amr_DEPENDENCIES 
      ${had_amr_DEPENDENCIES}
      ${SDF_LIBRARY})
endif()

add_hpx_executable(had_amr_client 
    SOURCES ${had_amr_client_SOURCES}
    DEPENDENCIES ${had_amr_DEPENDENCIES})

# add a dependency to the pseudo-target  
add_dependencies(examples.had_amr had_amr_client_exe) 


