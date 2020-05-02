# Copyright (c) 2007-2015 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

function(hpx_handle_component_dependencies components)
  set(tmp "")

  foreach(component ${${components}})
    if(TARGET ${component}_component)
      set(tmp ${tmp} ${component}_component)
    elseif(TARGET HPX::${component}_component)
      set(tmp ${tmp} HPX::${component}_component)
    else()
      set(tmp ${tmp} hpx_${component})
    endif()
    hpx_debug("hpx_handle_component_dependencies: ${tmp}")
  endforeach()

  set(${components}
      ${tmp}
      PARENT_SCOPE
  )
endfunction()
