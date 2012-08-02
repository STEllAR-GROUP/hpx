# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_FINDPROGRAM_LOADED TRUE)

if(NOT HPX_UTILS_LOADED)
  include(HPX_Utils)
endif()

macro(hpx_find_program name)
  if(${name}_DISABLE)
    hpx_info("find_program.${name}" "Program search disabled by user.")
    unset(${name}_ROOT)
    set(${name}_FOUND OFF CACHE BOOL "Found ${name}.")
    set(${name}_PROGRAM ${name}_PROGRAM-NOTFOUND CACHE FILEPATH "${name} program.")
    mark_as_advanced(FORCE ${name}_PROGRAM)
  else()
    if(NOT ${name}_SEARCHED)
      hpx_info("find_program.${name}" "Searching for program ${name}.")

      hpx_parse_arguments(${name} "PROGRAMS;PROGRAM_PATHS" "ESSENTIAL" ${ARGN})

      set(rooted_paths)
      foreach(path ${${name}_PROGRAM_PATHS})
        list(APPEND rooted_paths ${${name}_ROOT}/${path})
      endforeach()

      if(${name}_ROOT)
        hpx_print_list("DEBUG" "find_program.${name}" "Program names" ${name}_PROGRAMS)
        hpx_print_list("DEBUG" "find_program.${name}" "Program paths" rooted_paths)
        find_program(${name}_PROGRAM
          NAMES ${${name}_PROGRAMS}
          PATHS ${rooted_paths}
          NO_DEFAULT_PATH)

        if(NOT ${name}_PROGRAM)
          hpx_warn("find_program.${name}" "Program not found in ${${name}_ROOT}, trying system path.")
          unset(${name}_ROOT)
        else()
          set(${name}_SEARCHED ON CACHE INTERNAL "Searched for ${name} program.")
          hpx_info("find_program.${name}" "Program found in ${${name}_ROOT}.")
        endif()
      endif()

      # if not found, retry using system path
      if(NOT ${name}_ROOT)
        hpx_print_list("DEBUG" "find_program.${name}" "Program names" ${name}_PROGRAMS)
        hpx_print_list("DEBUG" "find_program.${name}" "Program paths" ${name}_PROGRAM_PATHS)
        find_program(${name}_PROGRAM NAMES ${${name}_PROGRAMS}
                                        PATH_SUFFIXES ${${name}_PROGRAM_PATHS})

        if(NOT ${name}_PROGRAM)
          if(${name}_ESSENTIAL)
            hpx_error("find_program.${name}" "Program not found in system path.")
          else()
            hpx_warn("find_program.${name}" "Program not found in system path.")
          endif()
          unset(${name}_SEARCHED CACHE)
          unset(${name}_ROOT)
        else()
          set(${name}_SEARCHED ON CACHE INTERNAL "Searched for ${name} program.")
          hpx_info("find_program.${name}" "Program found in system path.")
        endif()
      endif()

      if(${name}_PROGRAM)
        set(${name}_FOUND ON CACHE BOOL "Found ${name}." FORCE)
        set(${name}_ROOT ${${name}_ROOT} CACHE PATH "${name} root directory.")
        set(${name}_PROGRAM ${${name}_PROGRAM} CACHE FILEPATH "${name} program.")
        mark_as_advanced(FORCE ${name}_ROOT ${name}_PROGRAM)
      endif()
    endif()
  endif()
endmacro()

