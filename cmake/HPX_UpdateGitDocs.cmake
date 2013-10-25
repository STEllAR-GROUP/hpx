# Copyright (c) 2011-2013 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(Git)

if(NOT GIT_FOUND)
  message(FATAL_ERROR "Could not find git. git is needed to download and update the GitHub pages.")
endif()

if(EXISTS ${CMAKE_BINARY_DIR}/gh-pages)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} pull --rebase
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/gh-pages
    RESULT_VARIABLE git_pull_result)
  if(NOT "${git_pull_result}" EQUAL "0")
    message(FATAL_ERROR "Updating the GitHub pages branch failed.")
  endif()
else()
  execute_process(
    COMMAND ${GIT_EXECUTABLE} clone git@github.com:STEllAR-GROUP/hpx.git --branch gh-pages gh-pages
    RESULT_VARIABLE git_clone_result)
  if(NOT "${git_clone_result}" EQUAL "0")
    message(FATAL_ERROR "Cloning the GitHub pages branch failed.")
  endif()
endif()

# first delete all html files
file(REMOVE_RECURSE ${CMAKE_BINARY_DIR}/gh-pages/docs/html/hpx)

# copy all documentation files to target branch
file(
  COPY ${HPX_SOURCE_DIR}/docs/html
  DESTINATION ${CMAKE_BINARY_DIR}/gh-pages/docs)

if(HPX_BUILD_TYPE)
  set(doc_dir ${CMAKE_BINARY_DIR}/${HPX_BUILD_TYPE}/../share/hpx-${HPX_VERSION})
else()
  set(doc_dir ${CMAKE_BINARY_DIR}/../share/hpx-${HPX_VERSION})
endif()

# disable copying source files for now, this needs to be fixed...
file(
  COPY ${doc_dir}/docs
  DESTINATION ${CMAKE_BINARY_DIR}/gh-pages)

# copy all source files the docs depend upon
if(HPX_DOCUMENTATION_FILES)
  string(REPLACE " " ";" HPX_DOCUMENTATION_FILES_LIST "${HPX_DOCUMENTATION_FILES}")
  foreach(file ${HPX_DOCUMENTATION_FILES_LIST})
    file(COPY ${file}
      DESTINATION ${CMAKE_BINARY_DIR}/gh-pages/docs/html/code)
  endforeach()
endif()

# add all newly generated file
execute_process(
  COMMAND ${GIT_EXECUTABLE} add *
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/gh-pages
  RESULT_VARIABLE git_add_result)
if(NOT "${git_add_result}" EQUAL "0")
  message(FATAL_ERROR "Adding files to the GitHub pages branch failed.")
endif()

# check if there are changes to commit
execute_process(
  COMMAND ${GIT_EXECUTABLE} diff-index --quiet HEAD
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/gh-pages
  RESULT_VARIABLE git_diff_index_result)
if(NOT "${git_diff_index_result}" EQUAL "0")
  # commit changes
  execute_process(
    COMMAND ${GIT_EXECUTABLE} commit -am "Updating docs"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/gh-pages
    RESULT_VARIABLE git_commit_result)
  if(NOT "${git_commit_result}" EQUAL "0")
    message(FATAL_ERROR "Commiting to the GitHub pages branch failed.")
  endif()
  
  # push everything up to github
  execute_process(
    COMMAND ${GIT_EXECUTABLE} push
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/gh-pages
    RESULT_VARIABLE git_push_result)
  if(NOT "${git_push_result}" EQUAL "0")
    message(FATAL_ERROR "Pushing to the GitHub pages branch failed.")
  endif()
endif()
