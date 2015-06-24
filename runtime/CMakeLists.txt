# Copyright (c) 2007-2012 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_hpx_executable(hpx_runtime
    SOURCES hpx_runtime.cpp
    FOLDER "Runtime")

add_hpx_pseudo_dependencies(runtime hpx_runtime_exe)

