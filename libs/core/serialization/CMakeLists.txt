# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Compatibility with using Boost.FileSystem, introduced in V1.4.0
hpx_option(
  HPX_SERIALIZATION_WITH_BOOST_TYPES BOOL
  "Enable serialization of certain Boost types. (default: ON)" ON ADVANCED
  CATEGORY "Modules"
  MODULE SERIALIZATION
)

if(HPX_SERIALIZATION_WITH_BOOST_TYPES)
  hpx_add_config_define_namespace(
    DEFINE HPX_SERIALIZATION_HAVE_BOOST_TYPES NAMESPACE SERIALIZATION
  )
endif()

# Default location is $HPX_ROOT/libs/serialization/include
set(serialization_headers
    hpx/serialization.hpp
    hpx/serialization/detail/extra_archive_data.hpp
    hpx/serialization/detail/non_default_constructible.hpp
    hpx/serialization/detail/pointer.hpp
    hpx/serialization/detail/polymorphic_id_factory.hpp
    hpx/serialization/detail/polymorphic_intrusive_factory.hpp
    hpx/serialization/detail/polymorphic_nonintrusive_factory.hpp
    hpx/serialization/detail/polymorphic_nonintrusive_factory_impl.hpp
    hpx/serialization/detail/preprocess_container.hpp
    hpx/serialization/detail/raw_ptr.hpp
    hpx/serialization/detail/serialize_collection.hpp
    hpx/serialization/detail/vc.hpp
    hpx/serialization/array.hpp
    hpx/serialization/bitset.hpp
    hpx/serialization/complex.hpp
    hpx/serialization/datapar.hpp
    hpx/serialization/deque.hpp
    hpx/serialization/dynamic_bitset.hpp
    hpx/serialization/exception_ptr.hpp
    hpx/serialization/list.hpp
    hpx/serialization/map.hpp
    hpx/serialization/multi_array.hpp
    hpx/serialization/optional.hpp
    hpx/serialization/set.hpp
    hpx/serialization/serialize_buffer.hpp
    hpx/serialization/string.hpp
    hpx/serialization/std_tuple.hpp
    hpx/serialization/tuple.hpp
    hpx/serialization/unordered_map.hpp
    hpx/serialization/vector.hpp
    hpx/serialization/variant.hpp
    hpx/serialization/valarray.hpp
    hpx/serialization/intrusive_ptr.hpp
    hpx/serialization/shared_ptr.hpp
    hpx/serialization/unique_ptr.hpp
    hpx/serialization/access.hpp
    hpx/serialization/base_object.hpp
    hpx/serialization/basic_archive.hpp
    hpx/serialization/binary_filter.hpp
    hpx/serialization/brace_initializable.hpp
    hpx/serialization/brace_initializable_fwd.hpp
    hpx/serialization/container.hpp
    hpx/serialization/input_archive.hpp
    hpx/serialization/input_container.hpp
    hpx/serialization/output_archive.hpp
    hpx/serialization/output_container.hpp
    hpx/serialization/serializable_any.hpp
    hpx/serialization/serialization_chunk.hpp
    hpx/serialization/serialization_fwd.hpp
    hpx/serialization/serialize.hpp
    hpx/serialization/traits/brace_initializable_traits.hpp
    hpx/serialization/traits/is_bitwise_serializable.hpp
    hpx/serialization/traits/needs_automatic_registration.hpp
    hpx/serialization/traits/polymorphic_traits.hpp
    hpx/serialization/traits/serialization_access_data.hpp
)

if(HPX_SERIALIZATION_WITH_BOOST_TYPES)
  set(serialization_headers ${serialization_headers}
                            hpx/serialization/boost_variant.hpp
  )
endif()

# Default location is $HPX_ROOT/libs/serialization/include_compatibility
# cmake-format: off
set(serialization_compat_headers
    hpx/runtime/serialization/array.hpp => hpx/serialization.hpp
    hpx/runtime/serialization/bitset.hpp => hpx/serialization.hpp
    hpx/runtime/serialization/complex.hpp => hpx/serialization.hpp
    hpx/runtime/serialization/container.hpp => hpx/serialization.hpp
    hpx/runtime/serialization/datapar.hpp => hpx/serialization.hpp
    hpx/runtime/serialization/deque.hpp => hpx/serialization.hpp
    hpx/runtime/serialization/dynamic_bitset.hpp => hpx/serialization.hpp
    hpx/runtime/serialization/list.hpp => hpx/serialization.hpp
    hpx/runtime/serialization/map.hpp => hpx/serialization.hpp
    hpx/runtime/serialization/multi_array.hpp => hpx/serialization.hpp
    hpx/runtime/serialization/optional.hpp => hpx/serialization.hpp
    hpx/runtime/serialization/serialize_buffer.hpp => hpx/serialization.hpp
    hpx/runtime/serialization/set.hpp => hpx/serialization.hpp
    hpx/runtime/serialization/string.hpp => hpx/serialization.hpp
    hpx/runtime/serialization/tuple.hpp => hpx/serialization.hpp
    hpx/runtime/serialization/vector.hpp => hpx/serialization.hpp
    hpx/runtime/serialization/variant.hpp => hpx/serialization.hpp
    hpx/runtime/serialization/valarray.hpp => hpx/serialization.hpp
    hpx/runtime/serialization/intrusive_ptr.hpp => hpx/serialization.hpp
    hpx/runtime/serialization/shared_ptr.hpp => hpx/serialization.hpp
    hpx/runtime/serialization/unique_ptr.hpp => hpx/serialization.hpp
    hpx/runtime/serialization/unordered_map.hpp => hpx/serialization.hpp
    hpx/runtime/serialization/access.hpp => hpx/serialization.hpp
    hpx/runtime/serialization/base_object.hpp => hpx/serialization.hpp
    hpx/runtime/serialization/basic_archive.hpp => hpx/serialization.hpp
    hpx/runtime/serialization/binary_filter.hpp => hpx/serialization.hpp
    hpx/runtime/serialization/brace_initializable.hpp => hpx/serialization.hpp
    hpx/runtime/serialization/brace_initializable_fwd.hpp => hpx/serialization.hpp
    hpx/runtime/serialization/input_archive.hpp => hpx/serialization.hpp
    hpx/runtime/serialization/input_container.hpp => hpx/serialization.hpp
    hpx/runtime/serialization/output_archive.hpp => hpx/serialization.hpp
    hpx/runtime/serialization/output_container.hpp => hpx/serialization.hpp
    hpx/runtime/serialization/serialization_chunk.hpp => hpx/serialization.hpp
    hpx/runtime/serialization/serialization_fwd.hpp => hpx/serialization.hpp
    hpx/runtime/serialization/serialize.hpp => hpx/serialization.hpp
    hpx/traits/brace_initializable_traits.hpp => hpx/serialization.hpp
    hpx/traits/is_bitwise_serializable.hpp => hpx/serialization.hpp
    hpx/traits/needs_automatic_registration.hpp => hpx/serialization.hpp
    hpx/traits/polymorphic_traits.hpp => hpx/serialization.hpp
    hpx/traits/serialization_access_data.hpp => hpx/serialization.hpp
    hpx/util/serializable_any.hpp => hpx/serialization.hpp
)
# cmake-format: on

# Default location is $HPX_ROOT/libs/serialization/src
set(serialization_sources
    detail/pointer.cpp
    detail/polymorphic_id_factory.cpp
    detail/polymorphic_intrusive_factory.cpp
    detail/polymorphic_nonintrusive_factory.cpp
    exception_ptr.cpp
    serializable_any.cpp
)

if(TARGET Vc::vc)
  set(serialization_optional_dependencies Vc::vc)
endif()

include(HPX_AddModule)
add_hpx_module(
  core serialization
  COMPATIBILITY_HEADERS ON # Added in 1.5.0
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  SOURCES ${serialization_sources}
  HEADERS ${serialization_headers}
  COMPAT_HEADERS ${serialization_compat_headers}
  MODULE_DEPENDENCIES
    hpx_assertion
    hpx_config
    hpx_datastructures
    hpx_debugging
    hpx_errors
    hpx_format
    hpx_hashing
    hpx_preprocessor
    hpx_type_support
  DEPENDENCIES ${serialization_optional_dependencies}
  CMAKE_SUBDIRS examples tests
)
