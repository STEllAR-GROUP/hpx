# Copyright (c) 2013 Jeroen Habraken
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
if(WITH_SECURITY)
  #FIXME: find sodium etc. ...
# ################################################################################
# # Security, libsodium: P(ortable|ackageable) NaCl
# ################################################################################
# find_package(HPX_Sodium)
# if(SODIUM_FOUND)
#   hpx_add_config_define(HPX_HAVE_SODIUM)
#
#   hpx_include_sys_directories("${SODIUM_INCLUDE_DIR}")
#   hpx_link_sys_directories("${SODIUM_LIBRARY_DIR}")
#   if(MSVC)
#     hpx_add_config_define(SODIUM_STATIC)
#   endif()
#   set(hpx_RUNTIME_LIBRARIES ${hpx_RUNTIME_LIBRARIES} ${SODIUM_LIBRARY})
#
#   if(HPX_HAVE_SECURITY)
#     hpx_add_config_define(HPX_HAVE_SECURITY)
#   endif()
#
# elseif(HPX_HAVE_SECURITY)
#   hpx_error("security" "The libsodium library has to be available to enable security features.")
#   set(HPX_HAVE_SECURITY OFF)
# endif()

  set(root "${hpx_SOURCE_DIR}/hpx/components/security")

  add_hpx_component(security
    FOLDER "Core/Components"
    HEADER_ROOT ${root}
    AUTOGLOB
    ESSENTIAL)

  add_hpx_pseudo_dependencies(components.security
                              security_component)
endif()
