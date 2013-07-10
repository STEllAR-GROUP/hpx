# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
# Copyright (c) 2010-2011 Alexander Neundorf
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_PARSEARGUMENTS_LOADED TRUE)

include(HPX_Include)

hpx_include(ListContains)

macro(hpx_parse_arguments prefix arg_names option_names)
  set(DEFAULT_ARGS)

  foreach(arg_name ${arg_names})
    set(${prefix}_${arg_name})
  endforeach()

  foreach(option ${option_names})
    set(${prefix}_${option} FALSE)
  endforeach()

  set(current_arg_name DEFAULT_ARGS)
  set(current_arg_list)

  foreach(arg ${ARGN})
    hpx_list_contains(is_arg_name ${arg} ${arg_names})
    if(is_arg_name)
      set(${prefix}_${current_arg_name} ${current_arg_list})
      set(current_arg_name ${arg})
      set(current_arg_list)
    else()
      hpx_list_contains(is_option ${arg} ${option_names})
      if(is_option)
        set(${prefix}_${arg} TRUE)
      else()
        set(current_arg_list ${current_arg_list} ${arg})
      endif()
    endif()
  endforeach()

  foreach(arg ${arg_names})
      if(${arg} AND NOT ${prefix}_${arg})
        if(NOT ${prefix}_${arg})
          set(${prefix}_${arg} ${${arg}})
        endif()
      endif()
  endforeach()

  foreach(option ${option_names})
      if(${option} AND NOT ${prefix}_${arg})
        if(NOT ${prefix}_${option})
          set(${prefix}_${option} TRUE)
        endif()
      endif()
  endforeach()

  set(${prefix}_${current_arg_name} ${current_arg_list})
endmacro()

