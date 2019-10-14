//  Copyright (c) 2016-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TRAITS_VECTOR_PACK_LOAD_SEP_26_2016_0719PM)
#define HPX_PARALLEL_TRAITS_VECTOR_PACK_LOAD_SEP_26_2016_0719PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR)

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace traits {
    ///////////////////////////////////////////////////////////////////////////
    template <typename V, typename NewT>
    struct rebind_pack;

    ///////////////////////////////////////////////////////////////////////////
    template <typename V, typename ValueType, typename Enable = void>
    struct vector_pack_load;

    template <typename V, typename ValueType, typename Enable = void>
    struct vector_pack_store;
}}}    // namespace hpx::parallel::traits

#if !defined(__CUDACC__)
#include <hpx/parallel/traits/detail/boost_simd/vector_pack_load_store.hpp>
#include <hpx/parallel/traits/detail/vc/vector_pack_load_store.hpp>
#endif

#endif
#endif
