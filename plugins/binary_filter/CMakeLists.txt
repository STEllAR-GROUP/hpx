# Copyright (c) 2007-2013 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(binary_filter_plugins
    bzip2
    snappy
    zlib)

foreach(type ${binary_filter_plugins})
  add_hpx_pseudo_target(plugins.binary_filter.${type})
  add_subdirectory(${type})
  add_hpx_pseudo_dependencies(plugins.binary_filter plugins.binary_filter.${type})
endforeach()

macro(add_binary_filter_modules)
  add_bzip2_module()
  add_snappy_module()
  add_zlib_module()
endmacro()
