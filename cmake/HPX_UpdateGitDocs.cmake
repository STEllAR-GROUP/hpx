# Copyright (c) 2011-2013 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(Git)

if(NOT GIT_FOUND)
  message(FATAL_ERROR "Could not find git. git is needed to download and update the GitHub pages.")
endif()

if(NOT GIT_REPOSITORY)
  set(GIT_REPOSITORY git@github.com:STEllAR-GROUP/hpx.git --branch gh-pages)
endif()

if(EXISTS "${CMAKE_BINARY_DIR}/gh-pages")
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" pull --rebase
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/gh-pages"
    RESULT_VARIABLE git_pull_result)
  if(NOT "${git_pull_result}" EQUAL "0")
    message(FATAL_ERROR "Updating the GitHub pages branch failed.")
  endif()
else()
  message("${GIT_EXECUTABLE}")
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" clone ${GIT_REPOSITORY} gh-pages
    RESULT_VARIABLE git_clone_result)
  if(NOT "${git_clone_result}" EQUAL "0")
    message(FATAL_ERROR "Cloning the GitHub pages branch failed. Trying to clone ${GIT_REPOSITORY}")
  endif()
endif()

# first delete all html files
file(REMOVE_RECURSE "${CMAKE_BINARY_DIR}/gh-pages/docs/html/hpx")

# copy all documentation files to target branch
file(
    COPY "${HPX_SOURCE_DIR}/docs/html"
  DESTINATION "${CMAKE_BINARY_DIR}/gh-pages/docs")

set(doc_dir ${CMAKE_BINARY_DIR}/../share/hpx-${HPX_VERSION})

string(REPLACE "\"" "" doc_dir "${doc_dir}")

# Copy all documentation related files
file(
  COPY "${doc_dir}/docs"
  DESTINATION "${CMAKE_BINARY_DIR}/gh-pages"
  PATTERN "*code*" EXCLUDE
  PATTERN "*src*" EXCLUDE
  PATTERN "*images*" EXCLUDE)

# copy all source files the docs depend upon
if(HPX_DOCUMENTATION_FILES)
  string(REPLACE " " ";" HPX_DOCUMENTATION_FILES_LIST "${HPX_DOCUMENTATION_FILES}")
  foreach(file ${HPX_DOCUMENTATION_FILES_LIST})
    string(REPLACE "\"" "" file ${file})
    get_filename_component(dest "${file}" PATH)
    string(REPLACE "${HPX_SOURCE_DIR}/" "" dest ${dest})
    file(COPY "${file}"
      DESTINATION "${CMAKE_BINARY_DIR}/gh-pages/docs/html/code/${dest}")
  endforeach()
endif()

# add all newly generated file
execute_process(
  COMMAND "${GIT_EXECUTABLE}" add *
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/gh-pages"
  RESULT_VARIABLE git_add_result)
if(NOT "${git_add_result}" EQUAL "0")
  message(FATAL_ERROR "Adding files to the GitHub pages branch failed.")
endif()

# check if there are changes to commit
execute_process(
  COMMAND "${GIT_EXECUTABLE}" diff-index --quiet HEAD
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/gh-pages"
  RESULT_VARIABLE git_diff_index_result)
if(NOT "${git_diff_index_result}" EQUAL "0")
  # commit changes
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" commit -am "Updating docs"
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/gh-pages"
    RESULT_VARIABLE git_commit_result)
  if(NOT "${git_commit_result}" EQUAL "0")
    message(FATAL_ERROR "Commiting to the GitHub pages branch failed.")
  endif()

  # push everything up to github
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" push
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/gh-pages"
    RESULT_VARIABLE git_push_result)
  if(NOT "${git_push_result}" EQUAL "0")
    message(FATAL_ERROR "Pushing to the GitHub pages branch failed.")
 endif()
endif()
