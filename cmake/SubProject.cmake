# Copyright (c) 2014-2015 Raphael Dumusc
# Copyright (c) 2014-2015 Ahmet Bilgili
# Copyright (c) 2014-2015 Stefan Eileman
# Copyright (c) 2014-2015 John Biddiscombe
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Include this file in a top-level CMakeLists to build several CMake
# subprojects (which may depend on each other).
#
# When included, it will automatically parse a .gitsubprojects file if one is
# present in the same CMake source directory.
# .gitsubprojects contains lines in the form:
# "git_subproject(<project> <giturl> <gittag>)"
#
# All the subprojects will be cloned and configured during the CMake configure
# step thanks to the 'git_subproject(project giturl gittag)' macro
# (also usable separately).
# The latter relies on the add_subproject(name) function to add projects as
# sub directories. See also: cmake command 'add_subdirectory'.
#
# The following targets are created by SubProject.cmake:
# - An 'update_git_subprojects_${PROJECT_NAME}' target to update the <gittag> of
#   all the .gitsubprojects entries to their latest respective origin/master ref
# - A generic 'update' target to execute 'update_git_subrojects' recursively
#
# To be compatible with the SubProject feature, (sub)projects might need to
# adapt their CMake scripts in the following way:
# - CMAKE_BINARY_DIR should be changed to PROJECT_BINARY_DIR
# - CMAKE_SOURCE_DIR should be changed to PROJECT_SOURCE_DIR
#
# Respects the following variables:
# - DISABLE_SUBPROJECTS: when set, does not load sub projects. Useful for
#   example for continuous integration builds
# A sample project can be found at https://github.com/Eyescale/Collage.git

#include(${CMAKE_CURRENT_LIST_DIR}/GitExternal.cmake)
#include(${CMAKE_CURRENT_LIST_DIR}/CMakeCompatibility.cmake)

function(add_subproject name)
  string(TOUPPER ${name} NAME)
  if(CMAKE_MODULE_PATH)
    # We're adding a sub project here: Remove parent's CMake
    # directories so they don't take precendence over the sub project
    # directories. Change is scoped to this function.
    list(REMOVE_ITEM CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake
      ${PROJECT_SOURCE_DIR}/CMake/common)
  endif()

  list(LENGTH ARGN argc)
  if(argc GREATER 0)
    list(GET ARGN 0 path)
  else()
    set(path ${name})
  endif ()

  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${path}/")
    message(FATAL_ERROR "Sub project ${path} not found in ${CMAKE_CURRENT_SOURCE_DIR}")
  endif()

  option(SUBPROJECT_${name} "Build ${name} " ON)
  if(SUBPROJECT_${name})
    # if the project needs to do anything special when configured as a
    # sub project then it can check the variable ${PROJECT}_IS_SUBPROJECT
    set(${name}_IS_SUBPROJECT ON)
    set(${NAME}_FOUND ON PARENT_SCOPE)

    # set ${PROJECT}_DIR to the location of the new build dir for the project
    if(NOT ${name}_DIR)
      set(${name}_DIR "${CMAKE_BINARY_DIR}/${name}" CACHE PATH
        "Location of ${name} project" FORCE)
    endif()

    # add the source sub directory to our build and set the binary dir
    # to the build tree
    set(ADD_SUBPROJECT_INDENT "${ADD_SUBPROJECT_INDENT}   ")
    message("${ADD_SUBPROJECT_INDENT}========== ${path} ==========")
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/${path}"
      "${CMAKE_BINARY_DIR}/${name}")
    message("${ADD_SUBPROJECT_INDENT}---------- ${path} ----------")
    set(${name}_IS_SUBPROJECT ON PARENT_SCOPE)
    # Mark globally that we've already used name as a sub project
    set_property(GLOBAL PROPERTY ${name}_IS_SUBPROJECT ON)
  endif()
endfunction()

macro(git_subproject name url tag)
  if(NOT BUILDYARD AND NOT DISABLE_SUBPROJECTS)
    string(TOUPPER ${name} NAME)
    set(TAG ${tag})
    if(SUBPROJECT_TAG AND NOT "${tag}" STREQUAL "release")
      set(TAG ${SUBPROJECT_TAG})
    endif()
    if(NOT ${NAME}_FOUND)
      get_property(__included GLOBAL PROPERTY ${name}_IS_SUBPROJECT)
      if(NOT EXISTS ${CMAKE_CURRENT_CMAKE_CURRENT_SOURCE_DIRSOURCE_DIR}/${name})
        find_package(${name} QUIET CONFIG)
      elseif(__included) # already used as a sub project, just find it:
        find_package(${name} QUIET CONFIG HINTS ${CMAKE_BINARY_DIR}/${NAME})
      endif()
      if(NOT ${NAME}_FOUND)
        git_external(${CMAKE_CURRENT_SOURCE_DIR}/${name} ${url} ${TAG})
        add_subproject(${name})
        find_package(${name} REQUIRED CONFIG) # find subproject "package"
        include_directories(${${NAME}_INCLUDE_DIRS})
      endif()
    endif()
    get_property(__included GLOBAL PROPERTY ${name}_IS_SUBPROJECT)
    if(__included)
      list(APPEND __subprojects "${name} ${url} ${tag}")
    endif()
  endif()
endmacro()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.gitsubprojects")
  set(__subprojects) # appended on each git_subproject invocation
  include(.gitsubprojects)

  if(__subprojects)
    set(GIT_SUBPROJECTS_SCRIPT
      "${CMAKE_CURRENT_BINARY_DIR}/UpdateSubprojects.cmake")
    file(WRITE "${GIT_SUBPROJECTS_SCRIPT}"
      "file(WRITE .gitsubprojects \"# -*- mode: cmake -*-\n\")\n")
    foreach(__subproject ${__subprojects})
      string(REPLACE " " ";" __subproject_list ${__subproject})
      list(GET __subproject_list 0 __subproject_name)
      list(GET __subproject_list 1 __subproject_repo)
      set(__subproject_dir "${CMAKE_CURRENT_SOURCE_DIR}/${__subproject_name}")
      file(APPEND "${GIT_SUBPROJECTS_SCRIPT}"
        "execute_process(COMMAND ${GIT_EXECUTABLE} fetch origin -q\n"
        "  WORKING_DIRECTORY ${__subproject_dir})\n"
        "execute_process(COMMAND \n"
        "  ${GIT_EXECUTABLE} show-ref --hash=7 refs/remotes/origin/master\n"
        "  OUTPUT_VARIABLE newref OUTPUT_STRIP_TRAILING_WHITESPACE\n"
        "  WORKING_DIRECTORY ${__subproject_dir})\n"
        "if(newref)\n"
        "  file(APPEND .gitsubprojects\n"
        "    \"git_subproject(${__subproject_name} ${__subproject_repo} \${newref})\\n\")\n"
        "else()\n"
        "  file(APPEND .gitsubprojects \"git_subproject(${__subproject})\n\")\n"
        "endif()\n")
    endforeach()

    add_custom_target(update_git_subprojects_${PROJECT_NAME}
      COMMAND ${CMAKE_COMMAND} -P ${GIT_SUBPROJECTS_SCRIPT}
      COMMENT "Update ${PROJECT_NAME}/.gitsubprojects"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

    if(NOT TARGET update)
      add_custom_target(update)
    endif()
    add_dependencies(update update_git_subprojects_${PROJECT_NAME})
  endif()
endif()

