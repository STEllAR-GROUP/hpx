# Copyright (c) 2013 Jeroen Habraken
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
if(HPX_WITH_SECURITY)

  set(root "${PROJECT_SOURCE_DIR}/hpx/components/security")

  add_hpx_component(security
    FOLDER "Core/Components"
    HEADER_ROOT ${root}
    AUTOGLOB)

  add_hpx_pseudo_dependencies(components.security_ security_component)
  add_hpx_pseudo_dependencies(core components.security_)

endif()
