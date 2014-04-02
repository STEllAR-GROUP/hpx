# Copyright (c) 2007-2014 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################

add_hpx_executable(network_storage
  AUTOGLOB
  FOLDER "Benchmarks/Network/network_storage")

# add pseudo-target dependencies
add_hpx_pseudo_dependencies(
  tests.performance.network.network_storage
  network_storage_exe)

hpx_option(HPX_BENCHMARK_SCRIPTS_PATH PATH
  "Directory to place batch scripts in"
  ${hpx_BINARY_DIR}/scripts
  ADVANCED)

# our windows batch script needs to know the lib path for boost/hpx
# and the executable path of the network_storage test
if (WIN32)
  # use path to build dir here, not install dir
  file(TO_NATIVE_PATH "${HPX_BUILD_PREFIX}/bin/network_storage.exe" EXE_PATH)
  file(TO_NATIVE_PATH "${BOOST_LIBRARY_DIR};${HPX_PREFIX}/lib/hpx" LIB_PATH)

  #message("EXE_PATH is ${EXE_PATH}, LIB_PATH=${LIB_PATH}")

  # generate a simple batch file for convenience
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/network_storage.bat.in
    ${HPX_BENCHMARK_SCRIPTS_PATH}/network_storage.bat
    @ONLY)
else()
#-------------------------------------------------------
# Slurm script generator for network_storage benchmark
#
# We would like to generate a script which can be used to generate batch jobs
# to submit benchmarks to slurm using many combinations of setting.
#
# We add a custom command which takes our template script and
# expands out all the variables we need to pass into it.
# Unfortunately, due to the way cmake works, some variables are only known
# at build time, and not at cmake configure time. Using a custom command which
# calls cmake to run our script at build time, allows us to pass variables
# into the final script which is placed in our build dir.
#
#-------------------------------------------------------

# not required but can be useful sometimes for checking problems
  #set(LD_LIBRARY_PATH "${HPX_LIBRARY_OUTPUT_DIRECTORY_${CMAKE_CFG_INTDIR}}" CACHE PATH "Path to set when generating script tests")

# Make sure scripts dir exists
  execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${HPX_BENCHMARK_SCRIPTS_PATH})

#--------------------------------------------------
# Slurm script generator for
# Standalone Bandwidth test
#--------------------------------------------------
  set(SCRIPT_NAME "slurm-test-HPX-storage")
  ADD_CUSTOM_COMMAND(
      DEPENDS   ${CMAKE_CURRENT_SOURCE_DIR}/${SCRIPT_NAME}.sh.in
      COMMAND   ${CMAKE_COMMAND}
      ARGS      -DEXE_PATH="$<TARGET_FILE:network_storage_exe>"
                -DMPIEXEC="${MPIEXEC}"
                -DSCRIPT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
                -DSCRIPT_NAME=${SCRIPT_NAME}
                -DSCRIPT_DEST_DIR="${HPX_BENCHMARK_SCRIPTS_PATH}"
                -DLIB_PATH="${LD_LIBRARY_PATH}"
                -DJOB_OPTIONS1="${SLURM_JOB_OPTIONS1}"
                -P ${CMAKE_CURRENT_SOURCE_DIR}/copy_script.cmake
      OUTPUT    ${TESTING_SCRIPTS_PATH}/${SCRIPT_NAME}.sh
  )

  add_custom_target(script-network-storage
      ALL
      DEPENDS ${TESTING_SCRIPTS_PATH}/slurm-test-HPX-storage.sh
  )
endif()
