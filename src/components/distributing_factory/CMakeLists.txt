# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c)      2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
set(root "${hpx_SOURCE_DIR}/hpx/components/distributing_factory")

add_hpx_component(distributing_factory
  FOLDER "Core/Components"
  HEADER_ROOT ${root}
  AUTOGLOB)

add_hpx_pseudo_dependencies(components.distributing_factory
                            distributing_factory_component)

