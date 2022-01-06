# Copyright (c) 2007-2012 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_WITH_DISTRIBUTED_RUNTIME)
  return()
endif()

add_hpx_component(
  startup_shutdown INTERNAL_FLAGS NOEXPORT
  FOLDER "Examples/StartupShutdown"
  AUTOGLOB
)

# make pseudo-targets depend on master pseudo-target
add_hpx_pseudo_dependencies(
  examples.startup_shutdown startup_shutdown_component
)
