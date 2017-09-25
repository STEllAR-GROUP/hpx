//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARTITIONED_VECTOR_EXPORT_DEFINITIONS_AUG_02_2017_0658PM)
#define HPX_PARTITIONED_VECTOR_EXPORT_DEFINITIONS_AUG_02_2017_0658PM

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

#endif


