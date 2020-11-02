//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config/export_definitions.hpp>

#if defined(HPX_PARTITIONED_VECTOR_MODULE_EXPORTS)
# define HPX_PARTITIONED_VECTOR_EXPORT HPX_SYMBOL_EXPORT
#else
# define HPX_PARTITIONED_VECTOR_EXPORT HPX_SYMBOL_IMPORT
#endif

#if defined(HPX_GCC_VERSION) && !defined(HPX_CLANG_VERSION)
#define HPX_PARTITIONED_VECTOR_SPECIALIZATION_EXPORT HPX_PARTITIONED_VECTOR_EXPORT
#else
#define HPX_PARTITIONED_VECTOR_SPECIALIZATION_EXPORT
#endif



