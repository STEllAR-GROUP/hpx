# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_FINDFILE_LOADED TRUE)

if(NOT HPX_UTILS_LOADED)
  include(HPX_Utils)
endif()

macro(hpx_find_file name)
  if(${name}_DISABLE)
    hpx_info("find_file.${name}" "File search disabled by user.")
    unset(${name}_ROOT)
    set(${name}_FOUND OFF CACHE BOOL "Found ${name}.")
    set(${name}_FILE ${name}_FILE-NOTFOUND CACHE FILEPATH "${name} file.")
  else()
    if(NOT ${name}_SEARCHED)
      hpx_info("find_file.${name}" "Searching for file ${name}.")

      hpx_parse_arguments(${name} "FILES;FILE_PATHS" "ESSENTIAL" ${ARGN})

      set(rooted_paths)
      foreach(path ${${name}_FILE_PATHS})
        list(APPEND rooted_paths ${${name}_ROOT}/${path})
      endforeach()

      if(${name}_ROOT)
        hpx_print_list("DEBUG" "find_file.${name}" "File names" ${name}_FILES)
        hpx_print_list("DEBUG" "find_file.${name}" "File paths" rooted_paths)
        find_file(${name}_FILE
          NAMES ${${name}_FILES}
          PATHS ${rooted_paths}
          NO_DEFAULT_PATH)

        if(NOT ${name}_FILE)
          hpx_warn("find_file.${name}" "File not found in ${${name}_ROOT}, trying system path.")
          unset(${name}_ROOT)
        else()
          set(${name}_SEARCHED ON CACHE INTERNAL "Searched for ${name} file.")
          hpx_info("find_file.${name}" "File found in ${${name}_ROOT}.")
        endif()
      endif()

      # if not found, retry using system path
      if(NOT ${name}_ROOT)
        hpx_print_list("DEBUG" "find_file.${name}" "File names" ${name}_FILES)
        hpx_print_list("DEBUG" "find_file.${name}" "File paths" ${name}_FILE_PATHS)
        find_file(${name}_FILE NAMES ${${name}_FILES}
                               PATH_SUFFIXES ${${name}_FILE_PATHS})

        if(NOT ${name}_FILE)
          if(${name}_ESSENTIAL)
            hpx_error("find_file.${name}" "File not found in system path.")
          else()
            hpx_warn("find_file.${name}" "File not found in system path.")
          endif()
          unset(${name}_SEARCHED CACHE)
          unset(${name}_ROOT)
        else()
          set(${name}_SEARCHED ON CACHE INTERNAL "Searched for ${name} file.")
          hpx_info("find_file.${name}" "File found in system path.")
        endif()
      endif()

      if(${name}_FILE)
        set(${name}_FOUND ON CACHE BOOL "Found ${name}.")
        set(${name}_ROOT ${${name}_ROOT} CACHE PATH "${name} root directory.")
        set(${name}_FILE ${${name}_FILE} CACHE FILEPATH "${name} file.")
      endif()
    endif()
  endif()
endmacro()

