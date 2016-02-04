# Copyright (c) 2015      John Biddiscombe
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#-------------------------------------------------------------------------------
# adds hpx_ prefix to give hpx_${name} to libraries and components
#-------------------------------------------------------------------------------
MACRO (hpx_set_lib_name target name)
  # there is no need to change debug/release names explicitly
  # as we use CMAKE_DEBUG_POSTFIX to alter debug names

  hpx_debug("hpx_set_lib_name: target:" ${target} "name: " ${name})
  set_target_properties (${target}
      PROPERTIES
      OUTPUT_NAME                hpx_${name}
      DEBUG_OUTPUT_NAME          hpx_${name}
      RELEASE_OUTPUT_NAME        hpx_${name}
      MINSIZEREL_OUTPUT_NAME     hpx_${name}
      RELWITHDEBINFO_OUTPUT_NAME hpx_${name}
  )
ENDMACRO()
