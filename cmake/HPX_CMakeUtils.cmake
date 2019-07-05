# Copyright (c) 2019 Auriane Reverdell
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


# Prepend function cause not handle for cmake < 3.12
function(prepend res prefix)
  set(varList "")
  foreach(f ${ARGN})
    list(APPEND varList "${prefix}/${f}")
  endforeach(f)
  set(${res} "${varList}" PARENT_SCOPE)
endfunction(prepend)
