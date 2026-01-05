# Copyright (c) 2019-2026 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

hpx_option(
  HPX_IOSTREAMS_WITH_WIDE_STREAMS BOOL "Enable wide IO streams. (default: OFF)"
  OFF ADVANCED
  CATEGORY "Modules"
  MODULE IOSTREAMS
)

if(HPX_IOSTREAMS_WITH_WIDE_STREAMS)
  hpx_add_config_define_namespace(
    DEFINE HPX_IOSTREAMS_HAVE_WIDE_STREAMS NAMESPACE IOSTREAMS
  )
endif()

set(iostreams_headers
    hpx/iostreams/categories.hpp
    hpx/iostreams/chain.hpp
    hpx/iostreams/char_traits.hpp
    hpx/iostreams/checked_operations.hpp
    hpx/iostreams/close.hpp
    hpx/iostreams/combine.hpp
    hpx/iostreams/compose.hpp
    hpx/iostreams/concepts.hpp
    hpx/iostreams/constants.hpp
    hpx/iostreams/copy.hpp
    hpx/iostreams/detail/access_control.hpp
    hpx/iostreams/detail/adapter/concept_adapter.hpp
    hpx/iostreams/detail/adapter/direct_adapter.hpp
    hpx/iostreams/detail/adapter/mode_adapter.hpp
    hpx/iostreams/detail/adapter/non_blocking_adapter.hpp
    hpx/iostreams/detail/adapter/output_iterator_adapter.hpp
    hpx/iostreams/detail/adapter/range_adapter.hpp
    hpx/iostreams/detail/buffer.hpp
    hpx/iostreams/detail/dispatch.hpp
    hpx/iostreams/detail/double_object.hpp
    hpx/iostreams/detail/error.hpp
    hpx/iostreams/detail/execute.hpp
    hpx/iostreams/detail/functional.hpp
    hpx/iostreams/detail/resolve.hpp
    hpx/iostreams/detail/streambuf/chainbuf.hpp
    hpx/iostreams/detail/streambuf/direct_streambuf.hpp
    hpx/iostreams/detail/streambuf/indirect_streambuf.hpp
    hpx/iostreams/detail/streambuf/linked_streambuf.hpp
    hpx/iostreams/detail/wrap_unwrap.hpp
    hpx/iostreams/device/array.hpp
    hpx/iostreams/device/back_inserter.hpp
    hpx/iostreams/device/file.hpp
    hpx/iostreams/device/mapped_file.hpp
    hpx/iostreams/device/null.hpp
    hpx/iostreams/filter/symmetric.hpp
    hpx/iostreams/filtering_stream.hpp
    hpx/iostreams/filtering_streambuf.hpp
    hpx/iostreams/flush.hpp
    hpx/iostreams/imbue.hpp
    hpx/iostreams/input_sequence.hpp
    hpx/iostreams/operations_fwd.hpp
    hpx/iostreams/operations.hpp
    hpx/iostreams/optimal_buffer_size.hpp
    hpx/iostreams/output_sequence.hpp
    hpx/iostreams/pipeline.hpp
    hpx/iostreams/positioning.hpp
    hpx/iostreams/read.hpp
    hpx/iostreams/seek.hpp
    hpx/iostreams/stream.hpp
    hpx/iostreams/stream_buffer.hpp
    hpx/iostreams/traits.hpp
    hpx/iostreams/write.hpp
)

set(iostreams_sources)
set(additional_dependencies)
set(additional_include_directories)

if(HPX_WITH_COMPRESSION_BZIP2)
  set(iostreams_headers ${iostreams_headers} hpx/iostreams/filter/bzip2.hpp)
  set(iostreams_sources ${iostreams_sources} bzip2.cpp)

  find_package(BZip2 REQUIRED)
  set(additional_include_directories ${additional_include_directories}
                                     ${BZIP2_INCLUDE_DIR}
  )
  set(additional_dependencies ${additional_dependencies} ${BZIP2_LIBRARIES})
endif()

if(HPX_WITH_COMPRESSION_ZLIB)
  set(iostreams_headers ${iostreams_headers} hpx/iostreams/filter/zlib.hpp)
  set(iostreams_sources ${iostreams_sources} zlib.cpp)

  find_package(ZLIB REQUIRED)
  set(additional_include_directories ${additional_include_directories}
                                     ${ZLIB_INCLUDE_DIR}
  )
  set(additional_dependencies ${additional_dependencies} ${ZLIB_LIBRARIES})
endif()

include(HPX_AddModule)
add_hpx_module(
  core iostreams
  GLOBAL_HEADER_GEN ON
  SOURCES ${iostreams_sources}
  HEADERS ${iostreams_headers}
  COMPAT_HEADERS ${iostreams_compat_headers}
  DEPENDENCIES ${additional_dependencies}
  INCLUDE_DIRECTORIES ${additional_include_directories}
  MODULE_DEPENDENCIES hpx_config hpx_assertion hpx_datastructures hpx_functional
                      hpx_iterator_support hpx_type_support
  CMAKE_SUBDIRS tests
)
