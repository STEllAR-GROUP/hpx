# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_hpx_library_sources(hpx_serialization
  binary_portable_archives.cpp
  portable_binary_iarchive.cpp
  portable_binary_oarchive.cpp)

add_hpx_library_sources(hpx_ini
  ini.cpp)

add_hpx_library_sources(hpx
  init_ini_data.cpp
  itt_notify.cpp
  logging.cpp
  performance_counters.cpp
  runtime_configuration.cpp
  asio_util.cpp
  io_service_pool.cpp
  serialize_exception.cpp)

if("${HPX_AGAS_VERSION}" STREQUAL "2")
  add_hpx_library_sources(hpx generate_unique_ids_v2.cpp)
else()
  add_hpx_library_sources(hpx generate_unique_ids_v1.cpp)
endif()

