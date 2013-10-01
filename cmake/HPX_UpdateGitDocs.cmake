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
execute_process(
  COMMAND ${GIT_EXECUTABLE} rm *.html
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/gh-pages
  RESULT_VARIABLE git_rm_result)
if(NOT "${git_rm_result}" EQUAL "0")
    message(FATAL_ERROR "Removing stale html files from the GitHub pages branch failed.")
endif()

# copy all documentation files to target branch
file(
  COPY ${HPX_SOURCE_DIR}/docs/html
  DESTINATION ${CMAKE_BINARY_DIR}/gh-pages/docs)

file(
  COPY ${CMAKE_BINARY_DIR}/share/hpx/docs/html
  DESTINATION ${CMAKE_BINARY_DIR}/gh-pages/docs)

# add all newly generated file
execute_process(
  COMMAND ${GIT_EXECUTABLE} add *
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/gh-pages
  RESULT_VARIABLE git_add_result)
if(NOT "${git_add_result}" EQUAL "0")
    message(FATAL_ERROR "Adding files to the GitHub pages branch failed.")
endif()

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
