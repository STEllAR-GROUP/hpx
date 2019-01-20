# Copyright (c) 2018 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_library(hpx.pp.tests.unit.cat STATIC EXCLUDE_FROM_ALL cat.cpp)
add_library(hpx.pp.tests.unit.config STATIC EXCLUDE_FROM_ALL config.cpp)
add_library(hpx.pp.tests.unit.expand STATIC EXCLUDE_FROM_ALL expand.cpp)
add_library(hpx.pp.tests.unit.nargs STATIC EXCLUDE_FROM_ALL stringize.cpp)
add_library(hpx.pp.tests.unit.strip_parens STATIC EXCLUDE_FROM_ALL strip_parens.cpp)

target_link_libraries(hpx.pp.tests.unit.cat PRIVATE hpx.pp)
target_link_libraries(hpx.pp.tests.unit.config PRIVATE hpx.pp)
target_link_libraries(hpx.pp.tests.unit.expand PRIVATE hpx.pp)
target_link_libraries(hpx.pp.tests.unit.nargs PRIVATE hpx.pp)
target_link_libraries(hpx.pp.tests.unit.strip_parens PRIVATE hpx.pp)

add_dependencies(tests.unit.pp hpx.pp.tests.unit.cat)
add_dependencies(tests.unit.pp hpx.pp.tests.unit.config)
add_dependencies(tests.unit.pp hpx.pp.tests.unit.expand)
add_dependencies(tests.unit.pp hpx.pp.tests.unit.nargs)
add_dependencies(tests.unit.pp hpx.pp.tests.unit.strip_parens)
