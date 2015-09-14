# Copyright (c) 2014 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# If build of external build tests is disabled, exit now
if(NOT HPX_WITH_TESTS_EXTERNAL_BUILD)
  return()
endif()

# Try building an external cmake based project ...
macro(create_cmake_test name hpx_dir)
  set(build_dir "${CMAKE_CURRENT_BINARY_DIR}/${name}")
  add_custom_target(${name}.make_build_dir
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${build_dir}" VERBATIM
  )
  set(ADDITIONAL_CMAKE_OPTIONS)
  if(CMAKE_TOOLCHAIN_FILE)
    set(ADDITIONAL_CMAKE_OPTIONS -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE})
  endif()
  if(CMAKE_MAKE_COMMAND)
    set(ADDITIONAL_CMAKE_OPTIONS ${ADDITIONAL_CMAKE_OPTIONS} -DCMAKE_MAKE_COMMAND=${CMAKE_MAKE_COMMAND})
  endif()
  if(CMAKE_SYSROOT)
    set(ADDITIONAL_CMAKE_OPTIONS ${ADDITIONAL_CMAKE_OPTIONS} -DCMAKE_SYSROOT=${CMAKE_SYSROOT})
  endif()
  if(MSVC)
    set(ADDITIONAL_CMAKE_OPTIONS ${ADDITIONAL_CMAKE_OPTIONS} -G "${CMAKE_GENERATOR}")
  else()
    set(ADDITIONAL_CMAKE_OPTIONS ${ADDITIONAL_CMAKE_OPTIONS} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER})
    set(ADDITIONAL_CMAKE_OPTIONS ${ADDITIONAL_CMAKE_OPTIONS} -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER})
  endif()
  add_custom_target(${name}.make_configure
    COMMAND
      "${CMAKE_COMMAND}" -E chdir "${build_dir}" "${CMAKE_COMMAND}"
      "${PROJECT_SOURCE_DIR}/tests/unit/build/src"
      -DHPX_DIR=${hpx_dir}
      ${ADDITIONAL_CMAKE_OPTIONS}
      -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS_SAFE}
      -DCMAKE_BUILD_TYPE=$<CONFIGURATION> VERBATIM)
  add_dependencies(${name}.make_configure ${name}.make_build_dir hpx iostreams_component)
  if(NOT HPX_WITH_STATIC_LINKING)
    add_dependencies(${name}.make_configure hpx_init)
  endif()
  add_custom_target(${name}.make_compile
    COMMAND "${CMAKE_COMMAND}" --build "${build_dir}" VERBATIM
  )
  add_dependencies(${name}.make_compile ${name}.make_configure)
  add_hpx_pseudo_target(${name})
  add_hpx_pseudo_dependencies_no_shortening(${name} ${name}.make_compile)

  if(MSVC)
    set_target_properties(${name}.make_build_dir PROPERTIES
      FOLDER "Tests/Unit/Build")
    set_target_properties(${name}.make_configure PROPERTIES
      FOLDER "Tests/Unit/Build")
    set_target_properties(${name}.make_compile PROPERTIES
      FOLDER "Tests/Unit/Build")
  endif()
#  add_hpx_unit_test(
#    "build" ${name}
#    EXECUTABLE "${build_dir}/test"
#  )
endmacro()

macro(create_pkgconfig_test name hpx_dir)
  set(build_dir "${CMAKE_CURRENT_BINARY_DIR}/${name}")
  add_custom_target(${name}.make_build_dir
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${build_dir}" VERBATIM
  )
  add_custom_target(${name}.make_compile
    COMMAND
      "${CMAKE_COMMAND}" -E chdir "${build_dir}" make
      -f "${PROJECT_SOURCE_DIR}/tests/unit/build/src/Makefile"
      SRC_DIR=${PROJECT_SOURCE_DIR}/tests/unit/build/src
      HPX_DIR=${hpx_dir}
      CXX=${CMAKE_CXX_COMPILER}
      CXX_FLAGS=${CMAKE_CXX_FLAGS_SAFE}
      PKG_CONFIG=${PKG_CONFIG_EXECUTABLE}
      BUILD_TYPE=$<CONFIGURATION> VERBATIM)
  add_dependencies(${name}.make_compile ${name}.make_build_dir hpx iostreams_component)
  if(NOT HPX_WITH_STATIC_LINKING)
    add_dependencies(${name}.make_compile hpx_init)
  endif()
  add_hpx_pseudo_target(${name})
  add_hpx_pseudo_dependencies(${name} ${name}.make_compile)

#  add_hpx_unit_test(
#    "build" ${name}
#    EXECUTABLE "${build_dir}/test"
#  )
endmacro()

create_cmake_test(
  cmake_build_dir_test
  "${CMAKE_BINARY_DIR}/lib/cmake/${HPX_PACKAGE_NAME}")

create_cmake_test(
  cmake_install_dir_test
  "${CMAKE_INSTALL_PREFIX}/lib/cmake/${HPX_PACKAGE_NAME}"
)

find_package(PkgConfig)
if(PKGCONFIG_FOUND AND NOT MSVC)
  create_pkgconfig_test(
    pkgconfig_build_dir_test
    "${CMAKE_BINARY_DIR}/lib/pkgconfig")

  create_pkgconfig_test(
    pkgconfig_install_dir_test
    "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig")
endif()

set(tests
  build_dir
  install_dir)

set(build_systems cmake)
if(NOT CMAKE_TOOLCHAIN_FILE AND PKGCONFIG_FOUND AND NOT MSVC)
  set(build_systems ${build_systems} pkgconfig)
endif()

foreach(system ${build_systems})
  add_hpx_pseudo_target(tests.unit.build.${system})
  foreach(test ${tests})
    add_hpx_pseudo_target(tests.unit.build.${system}.${test})
    add_hpx_pseudo_dependencies(tests.unit.build.${system}.${test}
      ${system}_${test}_test)
    add_hpx_pseudo_dependencies(tests.unit.build.${system}
      tests.unit.build.${system}.${test})
  endforeach()

  add_hpx_pseudo_dependencies(tests.unit.build
    tests.unit.build.${system})
endforeach()
