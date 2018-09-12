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

set(SPHINX_DOCS_UNSTABLE_DEST "${CMAKE_BINARY_DIR}/gh-pages/docs/sphinx/unstable")
set(SPHINX_DOCS_VERSIONED_DEST "${CMAKE_BINARY_DIR}/gh-pages/docs/sphinx/${HPX_VERSION}")

# first delete all html files
file(REMOVE_RECURSE "${SPHINX_DOCS_UNSTABLE_DEST}")
file(REMOVE_RECURSE "${SPHINX_DOCS_VERSIONED_DEST}")

# copy all documentation files to target branch
set(SPHINX_DOCS_SOURCE "${HPX_BINARY_DIR}/share/hpx-${HPX_VERSION}/docs/html")
file(
  COPY "${SPHINX_DOCS_SOURCE}"
  DESTINATION "${SPHINX_DOCS_UNSTABLE_DEST}"
  PATTERN "*.buildinfo" EXCLUDE)
file(
  COPY "${SPHINX_DOCS_SOURCE}"
  DESTINATION "${SPHINX_DOCS_VERSIONED_DEST}"
  PATTERN "*.buildinfo" EXCLUDE)

# add all newly generated files
execute_process(
  COMMAND "${GIT_EXECUTABLE}" add *
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/gh-pages/docs/sphinx"
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
    COMMAND "${GIT_EXECUTABLE}" commit -am "Updating Sphinx docs"
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
