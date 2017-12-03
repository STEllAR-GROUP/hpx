#  Copyright (c) 2017 John Biddiscombe
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

#######################################################################
# For debugging this script
#######################################################################
message("Pull request is  " ${PYCICLE_PR})
message("PR-Branchname is " ${PYCICLE_BRANCH})
message("master branch is " ${PYCICLE_MASTER})
message("Machine name is  " ${PYCICLE_HOST})
message("PYCICLE_ROOT is  " ${PYCICLE_ROOT})
message("Random string is " ${PYCICLE_RANDOM})
message("COMPILER is      " ${PYCICLE_COMPILER})
message("BOOST is         " ${PYCICLE_BOOST})

#######################################################################
# Load machine specific settings
#######################################################################
include(${CMAKE_CURRENT_LIST_DIR}/config/${PYCICLE_HOST}.cmake)

#######################################################################
# User vars that need to be set on each machine using this script
#######################################################################
set(GIT_REPO "https://github.com/STEllAR-GROUP/hpx.git")

#######################################################################
# All the rest below here should not need changes
#######################################################################
set(PYCICLE_SRC_ROOT       "${PYCICLE_ROOT}/src")
set(PYCICLE_BUILD_ROOT     "${PYCICLE_ROOT}/build")
set(PYCICLE_LOCAL_GIT_COPY "${PYCICLE_ROOT}/repo")

if (PYCICLE_PR)
  set(CTEST_SOURCE_DIRECTORY "${PYCICLE_SRC_ROOT}/${PYCICLE_PR}/repo")
  set(CTEST_BINARY_DIRECTORY "${PYCICLE_BUILD_ROOT}/${PYCICLE_PR}")
  file(MAKE_DIRECTORY "${PYCICLE_SRC_ROOT}/${PYCICLE_PR}")
else()
  set(CTEST_SOURCE_DIRECTORY "${PYCICLE_SRC_ROOT}/master/repo")
  set(CTEST_BINARY_DIRECTORY "${PYCICLE_BUILD_ROOT}/master")
  file(MAKE_DIRECTORY "${PYCICLE_SRC_ROOT}/master")
endif()

set(CTEST_BUILD_CONFIGURATION "Release")
set(CTEST_BUILD_NAME "PR-${PYCICLE_PR}-${PYCICLE_BRANCH}-${CTEST_BUILD_CONFIGURATION}")

#######################################################################
# Not yet implemented memcheck/coverage/etc
#######################################################################
set(WITH_MEMCHECK FALSE)
set(WITH_COVERAGE FALSE)
if (WITH_MEMCHECK)
#  find_program(CTEST_COVERAGE_COMMAND NAMES gcov)
#  find_program(CTEST_MEMORYCHECK_COMMAND NAMES valgrind)
#  set(CTEST_MEMORYCHECK_SUPPRESSIONS_FILE ${CTEST_SOURCE_DIRECTORY}/tests/valgrind.supp)
endif()

#######################################################################
# Wipe build dir when starting a new build
#######################################################################
#ctest_empty_binary_directory(${CTEST_BINARY_DIRECTORY})

#######################################################################
# setup git
#######################################################################
include(FindGit)
set(CTEST_GIT_COMMAND "${GIT_EXECUTABLE}")

#######################################################################
# Initial checkout if no source directory
# if repo copy local - save time by copying instead of doing a clone
#######################################################################
if(NOT PYCICLE_LOCAL_GIT_COPY)
  message(FATAL_ERROR "You must have a local clone")
endif()

#######################################################################
# First checkout, copy from a local repo to save clone of many GB's
#######################################################################
set (make_repo_copy_ "")
if (NOT EXISTS "${CTEST_SOURCE_DIRECTORY}/.git")
  set (make_repo_copy_ "cp -r ${PYCICLE_LOCAL_GIT_COPY} ${CTEST_SOURCE_DIRECTORY};")
  message("cp -r ${PYCICLE_LOCAL_GIT_COPY} ${CTEST_SOURCE_DIRECTORY};")
endif()

#####################################################################
# if this is a PR to be merged with master for testing
#####################################################################
if (PYCICLE_PR)
  set(CTEST_SUBMISSION_TRACK "Pull Requests")
  set(PYCICLE_BRANCH "pull/${PYCICLE_PR}/head")
  set(GIT_BRANCH "PYCICLE_PR_${PYCICLE_PR}")
  #
  # checkout master, merge the PR into a new branch with the PR name
  # then checkout master again, then set the CTEST_UPDATE_OPTIONS
  # to fetch the merged branch so that the update step shows the
  # files that are different in the branch from master
  #
  set(WORK_DIR "${PYCICLE_SRC_ROOT}/${PYCICLE_PR}")
  execute_process(
    COMMAND bash "-c" "${make_repo_copy_}
                       cd ${CTEST_SOURCE_DIRECTORY};
                       ${CTEST_GIT_COMMAND} checkout ${PYCICLE_MASTER};
                       ${CTEST_GIT_COMMAND} fetch origin;
                       ${CTEST_GIT_COMMAND} reset --hard origin/${PYCICLE_MASTER};
                       ${CTEST_GIT_COMMAND} branch -D ${GIT_BRANCH};
                       ${CTEST_GIT_COMMAND} checkout -b ${GIT_BRANCH};
                       ${CTEST_GIT_COMMAND} fetch origin ${PYCICLE_BRANCH};
                       ${CTEST_GIT_COMMAND} merge --no-edit FETCH_HEAD;
                       ${CTEST_GIT_COMMAND} checkout ${PYCICLE_MASTER};
                       ${CTEST_GIT_COMMAND} clean -fd;"
    WORKING_DIRECTORY "${WORK_DIR}"
    OUTPUT_VARIABLE output
    ERROR_VARIABLE output
    RESULT_VARIABLE failed
  )
  set(CTEST_UPDATE_OPTIONS "${CTEST_SOURCE_DIRECTORY} ${GIT_BRANCH}")
else()
  set(CTEST_SUBMISSION_TRACK "Nightly")
  set(GIT_BRANCH "${PYCICLE_MASTER}")
  execute_process(
    COMMAND bash "-c" "${make_repo_copy_}
                       cd ${CTEST_SOURCE_DIRECTORY};
                       ${CTEST_GIT_COMMAND} checkout ${PYCICLE_MASTER};
                       ${CTEST_GIT_COMMAND} fetch origin;
                       ${CTEST_GIT_COMMAND} reset --hard;"
    WORKING_DIRECTORY "${WORK_DIR}"
    OUTPUT_VARIABLE output
    ERROR_VARIABLE output
    RESULT_VARIABLE failed
  )
  #message("Process output copy : " ${output})
endif()

#######################################################################
# Dashboard model : @TODO
#######################################################################
set(CTEST_MODEL Experimental)

#######################################################################
# START dashboard
#######################################################################
message("Initialize ${CTEST_MODEL} testing...")
ctest_start(${CTEST_MODEL}
    TRACK "${CTEST_SUBMISSION_TRACK}"
    "${CTEST_SOURCE_DIRECTORY}"
    "${CTEST_BINARY_DIRECTORY}"
)

#######################################################################
# Update dashboard
#######################################################################
message("Update source... using ${CTEST_SOURCE_DIRECTORY}")
ctest_update(RETURN_VALUE NB_CHANGED_FILES)
ctest_submit(PARTS Update)
message("Found ${NB_CHANGED_FILES} changed file(s)")

set(CTEST_CONFIGURE_COMMAND "${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE:STRING=${CTEST_BUILD_CONFIGURATION}")
set(CTEST_CONFIGURE_COMMAND "${CTEST_CONFIGURE_COMMAND} ${CTEST_BUILD_OPTIONS}")
set(CTEST_CONFIGURE_COMMAND "${CTEST_CONFIGURE_COMMAND} \"-G${CTEST_CMAKE_GENERATOR}\"")
set(CTEST_CONFIGURE_COMMAND "${CTEST_CONFIGURE_COMMAND} \"${CTEST_SOURCE_DIRECTORY}\"")

message("Configure...")
ctest_configure()
ctest_submit(PARTS Configure)

message("Build...")
ctest_build(TARGET "tests" FLAGS "-j ${BUILD_PARALLELISM}")
ctest_submit(PARTS Build)

message("Test...")
ctest_test(RETURN_VALUE retval EXCLUDE "compile")
ctest_submit(PARTS Test)

if (WITH_COVERAGE AND CTEST_COVERAGE_COMMAND)
  ctest_coverage()
endif (WITH_COVERAGE AND CTEST_COVERAGE_COMMAND)
if (WITH_MEMCHECK AND CTEST_MEMORYCHECK_COMMAND)
  ctest_memcheck()
endif (WITH_MEMCHECK AND CTEST_MEMORYCHECK_COMMAND)
