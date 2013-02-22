# Copyright (c) 2007-2013 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(plugin_categories
    compression)

foreach(category ${plugin_categories})
  add_hpx_pseudo_target(plugins.${category})
  add_subdirectory(${category})
  add_hpx_pseudo_dependencies(plugins plugins.${category})
endforeach()

macro(add_plugin_modules)
  add_compression_modules()
endmacro()
