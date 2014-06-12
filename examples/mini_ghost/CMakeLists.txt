# Copyright (c) 2012 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_HAVE_CXX11_LAMBDAS)
  #  set(mini_ghost_FLAGS
  #    DEPENDENCIES iostreams_component
  #                 mini_ghost_component
  #                 distributing_factory_component)
  add_definitions(-DHPX_LIMIT=10)

  # add mini_ghost executable
  add_hpx_executable(mini_ghost
    MODULE mini_ghost
    AUTOGLOB
    ${mini_ghost_FLAGS}
    FOLDER "Examples/MiniGhost")

  # add a custom target for this example
  add_hpx_pseudo_target(examples.mini_ghost.mini_ghost)

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.mini_ghost
                              examples.mini_ghost.mini_ghost)

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.mini_ghost.mini_ghost
                              mini_ghost_exe)

endif()
