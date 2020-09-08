# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(batch_environments_headers
    hpx/batch_environments/alps_environment.hpp
    hpx/batch_environments/batch_environment.hpp
    hpx/batch_environments/pbs_environment.hpp
    hpx/batch_environments/slurm_environment.hpp
)

set(batch_environments_compat_headers
    hpx/util/batch_environments/alps_environment.hpp
    hpx/util/batch_environment.hpp
    hpx/util/batch_environments/pbs_environment.hpp
    hpx/util/batch_environments/slurm_environment.hpp
)

set(batch_environments_sources alps_environment.cpp batch_environment.cpp
                               pbs_environment.cpp slurm_environment.cpp
)

include(HPX_AddModule)
add_hpx_module(
  full batch_environments
  COMPATIBILITY_HEADERS ON # Added in 1.5.0
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  SOURCES ${batch_environments_sources}
  HEADERS ${batch_environments_headers}
  COMPAT_HEADERS ${batch_environments_compat_headers}
  DEPENDENCIES hpx_core hpx_parallelism
  CMAKE_SUBDIRS examples tests
)
