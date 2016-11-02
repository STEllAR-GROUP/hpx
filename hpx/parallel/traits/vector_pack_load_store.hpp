//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TRAITS_VECTOR_PACK_LOAD_SEP_26_2016_0719PM)
#define HPX_PARALLEL_TRAITS_VECTOR_PACK_LOAD_SEP_26_2016_0719PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR)

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename V, typename NewT>
    struct rebind_pack;

    ///////////////////////////////////////////////////////////////////////////
    template <typename V, typename Enable = void>
    struct vector_pack_load;

    template <typename V, typename Enable = void>
    struct vector_pack_store;
}}}

#include <hpx/parallel/traits/detail/vc/vector_pack_load_store.hpp>
#include <hpx/parallel/traits/detail/boost_simd/vector_pack_load_store.hpp>

#endif
#endif

