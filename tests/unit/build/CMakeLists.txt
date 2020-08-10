# Copyright (c) 2014 Thomas Heller
# Copyright (c) 2018 Nikunj Gupta
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# If build of external build tests is disabled, exit now
if(NOT HPX_WITH_TESTS_EXTERNAL_BUILD)
  return()
endif()

if(NOT HPX_WITH_DISTRIBUTED_RUNTIME)
  return()
endif()

# Try building an external cmake based project ...
function(create_cmake_test name using_install_dir hpx_dir setup_type test_dir)
  set(build_dir "${CMAKE_CURRENT_BINARY_DIR}/${name}")
  add_custom_target(
    ${name}.make_build_dir
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${build_dir}"
    VERBATIM
  )
  set(ADDITIONAL_CMAKE_OPTIONS -DUSING_INSTALL_DIR=${using_install_dir})
  set(ADDITIONAL_CMAKE_OPTIONS -DSETUP_TYPE=${setup_type})
  if(CMAKE_TOOLCHAIN_FILE)
    set(ADDITIONAL_CMAKE_OPTIONS ${ADDITIONAL_CMAKE_OPTIONS}
                                 -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
    )
  endif()
  if(CMAKE_MAKE_COMMAND)
    set(ADDITIONAL_CMAKE_OPTIONS ${ADDITIONAL_CMAKE_OPTIONS}
                                 -DCMAKE_MAKE_COMMAND=${CMAKE_MAKE_COMMAND}
    )
  endif()
  if(CMAKE_SYSROOT)
    set(ADDITIONAL_CMAKE_OPTIONS ${ADDITIONAL_CMAKE_OPTIONS}
                                 -DCMAKE_SYSROOT=${CMAKE_SYSROOT}
    )
  endif()
  if(WIN32)
    set(ADDITIONAL_CMAKE_OPTIONS ${ADDITIONAL_CMAKE_OPTIONS} -G
                                 "${CMAKE_GENERATOR}"
    )
  endif()
  if(NOT MSVC)
    set(ADDITIONAL_CMAKE_OPTIONS ${ADDITIONAL_CMAKE_OPTIONS}
                                 -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    )
    set(ADDITIONAL_CMAKE_OPTIONS ${ADDITIONAL_CMAKE_OPTIONS}
                                 -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    )
  endif()
  set(test_dir "${PROJECT_SOURCE_DIR}/${test_dir}")
  add_custom_target(
    ${name}.make_configure
    COMMAND
      "${CMAKE_COMMAND}" -E chdir "${build_dir}" "${CMAKE_COMMAND}" ${test_dir}
      -DHPX_DIR=${hpx_dir} -DBOOST_ROOT=${BOOST_ROOT}
      ${ADDITIONAL_CMAKE_OPTIONS} -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS_SAFE}
      -DCMAKE_BUILD_TYPE=$<CONFIGURATION>
    VERBATIM
  )
  add_dependencies(
    ${name}.make_configure ${name}.make_build_dir hpx hpx_init
    iostreams_component
  )
  add_custom_target(
    ${name}.make_compile
    COMMAND "${CMAKE_COMMAND}" --build "${build_dir}" --config $<CONFIGURATION>
    VERBATIM
  )
  add_dependencies(${name}.make_compile ${name}.make_configure)
  add_hpx_pseudo_target(${name})
  add_hpx_pseudo_dependencies_no_shortening(${name} ${name}.make_compile)

  if(MSVC)
    set_target_properties(
      ${name}.make_build_dir PROPERTIES FOLDER "Tests/Unit/Build"
    )
    set_target_properties(
      ${name}.make_configure PROPERTIES FOLDER "Tests/Unit/Build"
    )
    set_target_properties(
      ${name}.make_compile PROPERTIES FOLDER "Tests/Unit/Build"
    )
  endif()
  # add_hpx_unit_test( "build" ${name} EXECUTABLE "${build_dir}/test" )
endfunction()

function(create_pkgconfig_test name hpx_dir)
  set(build_dir "${CMAKE_CURRENT_BINARY_DIR}/${name}")
  add_custom_target(
    ${name}.make_build_dir
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${build_dir}"
    VERBATIM
  )
  add_custom_target(
    ${name}.make_compile
    COMMAND
      "${CMAKE_COMMAND}" -E chdir "${build_dir}" make -f
      "${PROJECT_SOURCE_DIR}/examples/hello_world_component/Makefile"
      SRC_DIR=${PROJECT_SOURCE_DIR}/examples/hello_world_component
      HPX_DIR=${hpx_dir} CXX=${CMAKE_CXX_COMPILER}
      CXX_FLAGS=${CMAKE_CXX_FLAGS_SAFE} PKG_CONFIG=${PKG_CONFIG_EXECUTABLE}
      BUILD_TYPE=$<CONFIGURATION>
    VERBATIM
  )
  add_dependencies(
    ${name}.make_compile ${name}.make_build_dir hpx hpx_init
    iostreams_component
  )
  add_hpx_pseudo_target(${name})
  add_hpx_pseudo_dependencies(${name} ${name}.make_compile)

  # add_hpx_unit_test( "build" ${name} EXECUTABLE "${build_dir}/test" )
endfunction()

create_cmake_test(
  cmake_build_dir_targets_test FALSE
  "${PROJECT_BINARY_DIR}/lib/cmake/${HPX_PACKAGE_NAME}" TARGETS
  "examples/hello_world_component"
)

create_cmake_test(
  cmake_build_dir_macros_test FALSE
  "${PROJECT_BINARY_DIR}/lib/cmake/${HPX_PACKAGE_NAME}" MACROS
  "examples/hello_world_component"
)

create_cmake_test(
  cmake_install_dir_targets_test TRUE
  "${CMAKE_INSTALL_FULL_LIBDIR}/cmake/${HPX_PACKAGE_NAME}" TARGETS
  "examples/hello_world_component"
)

create_cmake_test(
  cmake_install_dir_macros_test TRUE
  "${CMAKE_INSTALL_FULL_LIBDIR}/cmake/${HPX_PACKAGE_NAME}" MACROS
  "examples/hello_world_component"
)

if(HPX_WITH_DYNAMIC_HPX_MAIN)
  # Google test emulation
  create_cmake_test(
    cmake_build_dir_gtest_emulation_test FALSE
    "${PROJECT_BINARY_DIR}/lib/cmake/${HPX_PACKAGE_NAME}" TARGETS
    "examples/gtest_emulation"
  )

  create_cmake_test(
    cmake_install_dir_gtest_emulation_test TRUE
    "${CMAKE_INSTALL_FULL_LIBDIR}/cmake/${HPX_PACKAGE_NAME}" TARGETS
    "examples/gtest_emulation"
  )
endif()

# PkgConfig
if(NOT MSVC)
  find_package(PkgConfig)
  if(PKGCONFIG_FOUND)
    create_pkgconfig_test(
      pkgconfig_build_dir_test "${PROJECT_BINARY_DIR}/lib/pkgconfig"
    )

    create_pkgconfig_test(
      pkgconfig_install_dir_test "${CMAKE_INSTALL_FULL_LIBDIR}/pkgconfig"
    )
  endif()
endif()

set(build_systems cmake)
set(cmake_tests build_dir_targets install_dir_targets build_dir_macros
                install_dir_macros
)
if(HPX_WITH_DYNAMIC_HPX_MAIN)
  list(APPEND cmake_tests build_dir_gtest_emulation install_dir_gtest_emulation)
endif()
if(NOT CMAKE_TOOLCHAIN_FILE
   AND PKGCONFIG_FOUND
   AND NOT MSVC
)
  set(build_systems ${build_systems} pkgconfig)
  set(pkgconfig_tests build_dir install_dir)
endif()

foreach(system ${build_systems})
  add_hpx_pseudo_target(tests.unit.build.${system})
  foreach(test ${${system}_tests})
    add_hpx_pseudo_target(tests.unit.build.${system}.${test})
    add_hpx_pseudo_dependencies(
      tests.unit.build.${system}.${test} ${system}_${test}_test
    )
    add_hpx_pseudo_dependencies(
      tests.unit.build.${system} tests.unit.build.${system}.${test}
    )
  endforeach()

  add_hpx_pseudo_dependencies(tests.unit.build tests.unit.build.${system})
endforeach()
