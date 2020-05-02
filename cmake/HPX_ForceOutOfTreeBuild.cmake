# Copyright (c) 2011 Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

function(hpx_force_out_of_tree_build message)
  string(COMPARE EQUAL "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}" insource)
  get_filename_component(parentdir ${PROJECT_SOURCE_DIR} PATH)
  string(COMPARE EQUAL "${PROJECT_SOURCE_DIR}" "${parentdir}" insourcesubdir)
  if(insource OR insourcesubdir)
    hpx_error("in_tree" "${message}")
  endif()
endfunction()
