# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_FINDFILE_LOADED TRUE)

if(NOT HPX_UTILS_LOADED)
  include(HPX_Utils)
endif()

macro(hpx_find_path name)
  if(${name}_DISABLE)
    hpx_info("find_path.${name}" "Path search disabled by user.")
    unset(${name}_ROOT)
    set(${name}_PATH_FOUND OFF CACHE BOOL "Found ${name} path.")
    set(${name}_PATH ${name}_PATH-NOTFOUND CACHE FILEPATH "${name} file.")
    mark_as_advanced(FORCE ${name}_PATH_FOUND ${name}_PATH})
  else()
    if(NOT ${name}_PATH_SEARCHED)
      hpx_info("find_path.${name}" "Searching for file ${name}.")

      hpx_parse_arguments(${name} "FILES;FILE_PATHS" "ESSENTIAL" ${ARGN})

      set(rooted_paths)
      foreach(path ${${name}_FILE_PATHS})
        list(APPEND rooted_paths ${${name}_ROOT}/${path})
      endforeach()

      if(${name}_ROOT)
        hpx_print_list("DEBUG" "find_path.${name}" "File names" ${name}_FILES)
        hpx_print_list("DEBUG" "find_path.${name}" "File paths" rooted_paths)
        find_path(${name}_PATH
          NAMES ${${name}_FILES}
          PATHS ${rooted_paths}
          NO_DEFAULT_PATH)

        if(NOT ${name}_PATH)
          hpx_warn("find_path.${name}" "File not found in ${${name}_ROOT}, trying system path.")
          unset(${name}_ROOT)
        else()
          set(${name}_PATH_SEARCHED ON CACHE INTERNAL "Searched for ${name} file.")
          hpx_info("find_path.${name}" "File found in ${${name}_ROOT}.")
        endif()
      endif()

      # if not found, retry using system path
      if(NOT ${name}_ROOT)
        hpx_print_list("DEBUG" "find_path.${name}" "File names" ${name}_FILES)
        hpx_print_list("DEBUG" "find_path.${name}" "File paths" ${name}_FILE_PATHS)
        find_path(${name}_PATH NAMES ${${name}_FILES}
                               PATH_SUFFIXES ${${name}_FILE_PATHS})

        if(NOT ${name}_PATH)
          if(${name}_ESSENTIAL)
            hpx_error("find_path.${name}" "File not found in system path.")
          else()
            hpx_warn("find_path.${name}" "File not found in system path.")
          endif()
          unset(${name}_PATH_SEARCHED CACHE)
          unset(${name}_ROOT)
        else()
          set(${name}_PATH_SEARCHED ON CACHE INTERNAL "Searched for ${name} file.")
          hpx_info("find_path.${name}" "File found in system path.")
        endif()
      endif()

      if(${name}_PATH)
        set(${name}_PATH_FOUND ON CACHE BOOL "Found ${name} path.")
        set(${name}_ROOT ${${name}_ROOT} CACHE PATH "${name} root directory.")
        set(${name}_PATH ${${name}_FILE} CACHE FILEPATH "Path to ${name} file.")
        mark_as_advanced(FORCE ${name}_ROOT ${name}_PATH)
      endif()
    endif()
  endif()
endmacro()

