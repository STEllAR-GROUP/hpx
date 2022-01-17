# Copyright (c) 2012 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_hpx_component(
  jacobi INTERNAL_FLAGS NOEXPORT AUTOGLOB
  SOURCE_GLOB "*.c*"
  HEADER_GLOB "*.h*" ${jacobi_component_FLAGS}
  FOLDER "Examples/Jacobi"
  COMPONENT_DEPENDENCIES iostreams
)
