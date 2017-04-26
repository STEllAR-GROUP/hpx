//  Copyright (c) 2016-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TRAITS_VECTOR_PACK_TYPE_OCT_31_2016_1232PM)
#define HPX_PARALLEL_TRAITS_VECTOR_PACK_TYPE_OCT_31_2016_1232PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR)
#include <hpx/util/tuple.hpp>

#include <cstddef>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    // exposition only
    template <typename T, std::size_t N = 0, typename Abi = void>
    struct vector_pack_type;

    // handle tuple<> transformations
    template <typename ... T, std::size_t N, typename Abi>
    struct vector_pack_type<hpx::util::tuple<T...>, N, Abi>
    {
        typedef hpx::util::tuple<
                typename vector_pack_type<T, N, Abi>::type...
            > type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename NewT>
    struct rebind_pack
    {
        typedef typename vector_pack_type<T>::type type;
    };
}}}

#if !defined(__CUDACC__)
#include <hpx/parallel/traits/detail/vc/vector_pack_type.hpp>
#include <hpx/parallel/traits/detail/boost_simd/vector_pack_type.hpp>
#endif

#endif
#endif

