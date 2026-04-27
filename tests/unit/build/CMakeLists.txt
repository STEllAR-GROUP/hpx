# Copyright (c) 2014 Thomas Heller
# Copyright (c) 2018 Nikunj Gupta
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# if the external build tests is disabled, exit now
if(NOT HPX_WITH_TESTS_EXTERNAL_BUILD)
  return()
endif()

# Try building an external cmake based project ...
function(
  create_cmake_test
  name
  using_install_dir
  hpx_dir
  setup_type
  build_type
)
  set(test_name "tests.unit.build.${name}")

  set(target_dir "${HPX_SOURCE_DIR}/examples/hello_world_component")
  set(source_dir "${PROJECT_BINARY_DIR}/tests/unit/build/${name}")

  file(MAKE_DIRECTORY "${source_dir}")

  # Check if we should use C++ modules
  set(modules_option "")
  if(HPX_WITH_CXX_MODULES)
    set(modules_option "-DHPX_WITH_CXX_MODULES=ON")
  endif()

  add_test(
    NAME "${test_name}.configure"
    COMMAND
      "${CMAKE_COMMAND}" "-H${target_dir}" "-B${source_dir}"
      "-DHPX_DIR=${hpx_dir}" "-DCMAKE_BUILD_TYPE=${build_type}"
      "-DHPX_WITH_MALLOC=${HPX_WITH_MALLOC}" "-G${CMAKE_GENERATOR}"
      "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
      "-DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}"
      "-DCMAKE_CXX_STANDARD_REQUIRED=${CMAKE_CXX_STANDARD_REQUIRED}"
      "-DCMAKE_CXX_EXTENSIONS=${CMAKE_CXX_STANDARD_EXTENSIONS}"
      "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
      "-DSETUP_TYPE=${setup_type}"
      "${modules_option}"
  )

  add_test(
    NAME "${test_name}.build"
    COMMAND "${CMAKE_COMMAND}" --build "${source_dir}" --config "${build_type}"
  )

  set_tests_properties(
    "${test_name}.build" PROPERTIES DEPENDS "${test_name}.configure"
  )

  add_hpx_pseudo_target("${test_name}")
  add_hpx_pseudo_dependencies("${test_name}" "${test_name}.build")
  add_hpx_pseudo_dependencies(tests.unit.build "${test_name}")
endfunction()

set(build_types "Debug")
if(NOT MSVC)
  set(build_types ${build_types} "Release")
endif()

foreach(build_type ${build_types})
  string(TOLOWER "${build_type}" build_type_lc)

  create_cmake_test(
    "cmake_dir_${build_type_lc}_test" FALSE
    "${PROJECT_BINARY_DIR}/lib/cmake/${HPX_PACKAGE_NAME}" "DIR" "${build_type}"
  )
endforeach()

if(HPX_WITH_PKGCONFIG
   AND HPX_WITH_DISTRIBUTED_RUNTIME
   AND NOT HPX_WITH_CXX_MODULES
)
  find_package(PkgConfig)
  if(PKGCONFIG_FOUND)
    create_pkgconfig_test(
      pkgconfig_build_dir_test "${PROJECT_BINARY_DIR}/lib/pkgconfig"
    )
    add_hpx_pseudo_dependencies(tests.unit.build tests.unit.build.pkgconfig)
  endif()
endif()
