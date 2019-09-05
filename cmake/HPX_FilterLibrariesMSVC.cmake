# Copyright (c) 2019 Ste||ar Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
# To solve a cmake error when invoking set_property with
# set_property(INTERFACE_LINK_LIBRARIES), this error doesn't appear with
# target_link_libraries(INTERFACE) but this function is not supported for
# imported libraries with CMake < 3.11
# example of the error:
# https://github.com/PointCloudLibrary/pcl/issues/2989
# So we use the following workaround, which can be removed as soon as we upgrade
# our required cmake version to 3.11

function(parse_libraries libraries name)
  if (MSVC)
    # Parse MSVC libraries to avoid cmake bug
    if ("${CMAKE_BUILD_TYPE}" MATCHES "Rel")
      set(_release true)
    else()
      set(_debug true)
    endif()
    foreach (lib IN LISTS libraries)
      # We add the current process lib if it has a flag and it matches the
      # build_type or if it has no flags
      if ("${lib}" MATCHES "^(optimized|general|debug)$")
        set(_build_flag_specified true)
      endif()
      if (NOT _build_flag_specified AND (NOT "${lib}" MATCHES "^(optimized|general|debug)$"))
        # If this is a basic library/target
        set(_output_libraries ${_output_libraries} ${lib})
      endif()
      if ((_release AND ("${lib}" MATCHES "^(optimized|general)$")) OR (_debug AND ("${lib}" MATCHES "^(debug|general)$")))
        set(_get_next true)
      else()
        if (_get_next)
          set(_output_libraries ${_output_libraries} ${lib})
          set(_get_next false)
          set(_build_flag_specified false)
        endif()
      endif()
    endforeach()
    set(${name} "${_output_libraries}" PARENT_SCOPE)
  else()
    set(${name} "${libraries}" PARENT_SCOPE)
  endif()
endfunction(parse_libraries)

# In order to fix the cmake bug, in case interface_link_libraries, we filter the
# libraries and then we call the old set_property with the filtered libraries
function(set_property)
  # Parse arguments
  set(options APPEND PROPERTY)
  set(one_value_args TARGET)
  set(multi_value_args INTERFACE_LINK_LIBRARIES)
  cmake_parse_arguments(my_props "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
  if (NOT my_props_INTERFACE_LINK_LIBRARIES)
    # No bug, we directly call the old function
    _set_property(${ARGN})
  else()
    parse_libraries("${my_props_INTERFACE_LINK_LIBRARIES}" filtered_libraries)
    if (my_props_APPEND)
      set(_append APPEND)
    endif()
    _set_property(TARGET ${my_props_TARGET} ${_append} PROPERTY INTERFACE_LINK_LIBRARIES ${filtered_libraries})
  endif()
endfunction(set_property)
