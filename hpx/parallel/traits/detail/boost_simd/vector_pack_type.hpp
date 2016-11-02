//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TRAITS_VECTOR_PACK_TYPE_BOOST_SIMD_OCT_31_2016_1229PM)
#define HPX_PARALLEL_TRAITS_VECTOR_PACK_TYPE_BOOST_SIMD_OCT_31_2016_1229PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR_BOOST_SIMD)
#include <hpx/util/tuple.hpp>

#include <cstddef>

#include <boost/simd.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace traits
{
    template <typename T,
        std::size_t N = boost::simd::native_cardinal<T>::value,
        typename Abi = boost::simd::abi_of_t<T, N> >
    struct vector_pack_type
    {
        typedef boost::simd::pack<T, N, Abi> type;
    };

    template <typename ... T, std::size_t N, typename Abi>
    struct vector_pack_type<hpx::util::tuple<T...>, N, Abi>
    {
        typedef hpx::util::tuple<boost::simd::pack<T, N, Abi>...> type;
    };
}}}

#endif
#endif

