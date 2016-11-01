//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TRAITS_VECTOR_PACK_ALIGNMENT_SIZE_BOOST_SIMD_SEP_29_2016_0905PM)
#define HPX_PARALLEL_TRAITS_VECTOR_PACK_ALIGNMENT_SIZE_BOOST_SIMD_SEP_29_2016_0905PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR_BOOST_SIMD)

#include <cstddef>
#include <type_traits>

#include <boost/simd.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename T, typename Enable>
    struct vector_pack_alignment
    {
        static std::size_t const value = boost::simd::pack<T>::alignment;
    };

    template <typename Iter, typename T, std::size_t N, typename Abi>
    struct vector_pack_alignment<Iter, boost::simd::pack<T, N, Abi> >
    {
        static std::size_t const value = boost::simd::pack<T, N, Abi>::alignment;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename T, typename Enable>
    struct vector_pack_size
    {
        static std::size_t const value = boost::simd::pack<T>::static_size;
    };

    template <typename Iter, typename T, std::size_t N, typename Abi>
    struct vector_pack_size<Iter, boost::simd::pack<T, N, Abi> >
    {
        static std::size_t const value = boost::simd::pack<T, N, Abi>::static_size;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, std::size_t N, typename Abi>
    struct vector_pack_is_scalar<boost::simd::pack<T, N, Abi> >
      : std::false_type
    {};

    template <typename T, typename Abi>
    struct vector_pack_is_scalar<boost::simd::pack<T, 1, Abi> >
      : std::true_type
    {};
}}}

#endif
#endif

