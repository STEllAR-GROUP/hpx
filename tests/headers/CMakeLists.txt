# Copyright (c) 2015-2017 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# collect a list of all header files in the source tree
file(GLOB_RECURSE headers "${PROJECT_SOURCE_DIR}/hpx/*hpp")

set(all_headers)

set(exclude_from_all_headers
  "util/hardware/timestamp/msvc.hpp"
  "util/hardware/timestamp/linux_x86_32.hpp"
  "util/hardware/timestamp/linux_x86_64.hpp"
  "util/hardware/timestamp/linux_generic.hpp"
  "util/hardware/timestamp/bgq.hpp"
)

set(exclude_from_headers
  "parallel/executors/execution_fwd.hpp"
  "parallel/executors/customization_point.hpp"
)

# If we compile with the MPI parcelport enabled, we need to additionally
# add the MPI include path here, because for the main library, it's only
# added for the plugin.
if(HPX_WITH_PARCELPORT_MPI)
  find_package(MPI)
  if(MPI_CXX_INCLUDE_PATH)
    include_directories(${MPI_CXX_INCLUDE_PATH})
  endif()
endif()

# If we compile with PAPI performance counters, we need to additionally
# add the PAPI include path here
if(HPX_WITH_PAPI)
  find_package(PAPI)
  include_directories(${PAPI_INCLUDE_DIR})
endif()

# make sure #including 'hpx/config/autolink.hpp' does not cause errors
hpx_add_compile_flag(-DHPX_AUTOLINK_LIB_NAME=hpx)

# for each of the found headers, generate a test executable
foreach(header ${headers})

  # skip all headers in directories containing 'detail'
  set(detail_pos -1)
  string(FIND "${header}" "detail" detail_pos)

  if(${detail_pos} EQUAL -1)
    # extract relative path of header
    string(REGEX REPLACE "${PROJECT_SOURCE_DIR}/hpx/" "" relpath "${header}")

    # .hpp --> .cpp
    string(REGEX REPLACE ".hpp" ".cpp" full_test_file "${relpath}")

    # exclude marked files
    set(detail_pos -1)
    list(FIND exclude_from_headers "${relpath}" detail_pos)
    if(${detail_pos} EQUAL -1)

      # remove extension, '/' --> '_'
      string(REGEX REPLACE ".hpp" "_hpp" test_file "${relpath}")
      string(REGEX REPLACE "/" "_" test_name "${test_file}")

      # generate the test
      file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${full_test_file}
        "#include <hpx/${relpath}>\n"
        "#ifndef HPX_MAIN_DEFINED\n"
        "int main(int argc, char** argv) { return 0; }\n"
        "#endif\n")

      # exclude from all files
      set(detail_pos -1)
      list(FIND exclude_from_all_headers "${relpath}" detail_pos)
      if(${detail_pos} EQUAL -1)
        set(all_headers ${all_headers} "#include <hpx/${relpath}>\n")
      endif()

      get_filename_component(header_dir "${relpath}" DIRECTORY)

      # add compile only test
      add_hpx_headers_compile_test(
        "headers"
        ${test_name}
        SOURCES "${CMAKE_CURRENT_BINARY_DIR}/${full_test_file}"
        SOURCE_ROOT "${CMAKE_CURRENT_BINARY_DIR}/${header_dir}"
        FOLDER "Tests/Headers/${header_dir}")

      # add a custom target for this example
      add_hpx_pseudo_target(tests.headers.${test_name})

      # make pseudo-targets depend on master pseudo-target
      add_hpx_pseudo_dependencies(tests.headers tests.headers.${test_name})

      # add dependencies to pseudo-target
      add_hpx_pseudo_dependencies(tests.headers.${test_name} "${test_name}_lib")
    endif()

  endif()

endforeach()

# generate a test executable project including all found header files
set(test_name "all_headers")
set(all_headers_test_name "${CMAKE_CURRENT_BINARY_DIR}/${test_name}.cpp")
file(WRITE ${all_headers_test_name}
  ${all_headers}
  "#ifndef HPX_MAIN_DEFINED\n"
  "int main(int argc, char** argv) { return 0; }\n"
  "#endif\n")

add_hpx_headers_compile_test(
  "all_headers"
  ${test_name}
  SOURCES "${all_headers_test_name}"
  SOURCE_ROOT "${CMAKE_CURRENT_BINARY_DIR}"
  FOLDER "Tests/Headers")

# add a custom target for this example
add_hpx_pseudo_target(tests.headers.${test_name})

# make pseudo-targets depend on master pseudo-target
add_hpx_pseudo_dependencies(tests.headers tests.headers.${test_name})

# add dependencies to pseudo-target
add_hpx_pseudo_dependencies(tests.headers.${test_name} "${test_name}_lib")
