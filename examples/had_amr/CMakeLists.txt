# Copyright (c) 2007-2010 Hartmut Kaiser
# Copyright (c) 2009-2010 Matt Anderson
# Copyright (c) 2011      Bryce Lelbach 
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# just recurse into all components subdirectories and execute the scripts there
set(subdirs 
    amr
    amr_c
    amr_c_test)

find_package(HPX_GMP)
find_package(HPX_RNPL)

if(RNPL_FOUND)
  # FIXME: HPX_* identifiers please
  add_definitions(-DRNPL_FOUND)
  include_directories(${RNPL_INCLUDE_DIR})
endif()

if(GMP_FOUND)
  # FIXME: HPX_* identifiers please
  add_definitions(-DGMP_FOUND)
  include_directories(${GMP_INCLUDE_DIR})

  # MPFR depends on GMP, so no point looking for it if we don't have GMP  
  find_package(HPX_MPFR)
  
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

foreach(subdir ${subdirs})
  add_subdirectory(${subdir})
endforeach()

# define build target for this directory
set(sources 
    amr_client.cpp)

# add these MPFR specific files if we have MPFR
if(MPFR_FOUND) 
  set(sources 
      ${sources}
      mpreal.cpp 
      serialize_mpreal.cpp)
endif()

# define basic dependencies
set(dependencies
    had_amr_component 
    had_amr_c_component 
    had_amr_c_test_component
    distributing_factory_component)

if(GMP_FOUND)
  # add GMP as a dependency
  set(dependencies 
      ${dependencies}
      ${GMP_LIBRARY})
  if(MPFR_FOUND)
    # add MPFR as a dependency
    set(dependencies 
        ${dependencies}
        ${MPFR_LIBRARY})
  endif()
endif()

if(RNPL_FOUND)
  # add RNPL as a dependency
  set(dependencies 
      ${dependencies}
      ${RNPL_LIBRARY})
endif()

source_group("Source Files" FILES ${sources})

add_hpx_executable(had_amr_client 
  MODULE had_amr
  SOURCES ${sources}
  DEPENDENCIES ${dependencies})

# add a dependency to the pseudo-target  
add_dependencies(examples.had_amr had_amr_client_exe had_amr_client) 

