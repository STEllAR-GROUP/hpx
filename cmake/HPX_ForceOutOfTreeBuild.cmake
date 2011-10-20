# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_FORCEOUTOFTREEBUILD_LOADED TRUE)

include(HPX_Include)

hpx_include(Message)

macro(hpx_force_out_of_tree_build message)
  string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}" insource)
  get_filename_component(parentdir ${CMAKE_SOURCE_DIR} PATH)
  string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${parentdir}" insourcesubdir)
  if(insource OR insourcesubdir)
    hpx_error("in_tree" "${message}")
  endif()
endmacro()

