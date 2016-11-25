//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TRAITS_VECTOR_PACK_LOAD_BOOST_SIMD_SEP_26_2016_0719PM)
#define HPX_PARALLEL_TRAITS_VECTOR_PACK_LOAD_BOOST_SIMD_SEP_26_2016_0719PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR_BOOST_SIMD)
#include <hpx/util/tuple.hpp>

#include <cstddef>
#include <iterator>
#include <memory>

#include <boost/simd.hpp>
#include <boost/simd/function/aligned_load.hpp>
#include <boost/simd/function/aligned_store.hpp>
#include <boost/simd/function/load.hpp>
#include <boost/simd/function/store.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T, std::size_t N, typename Abi, typename NewT>
    struct rebind_pack<boost::simd::pack<T, N, Abi>, NewT>
    {
        typedef boost::simd::pack<NewT, N, Abi> type;
    };

    // don't wrap types twice
    template <typename T, std::size_t N1, typename Abi1,
        typename NewT, std::size_t N2, typename Abi2>
    struct rebind_pack<boost::simd::pack<T, N1, Abi1>,
        boost::simd::pack<NewT, N2, Abi2> >
    {
        typedef boost::simd::pack<NewT, N2, Abi2> type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename V, typename ValueType, typename Enable>
    struct vector_pack_load
    {
        typedef typename rebind_pack<V, ValueType>::type value_type;

        template <typename Iter>
        static value_type aligned(Iter const& iter)
        {
            return boost::simd::aligned_load<vector_pack_type>(
                std::addressof(*iter));
        }

        template <typename Iter>
        static value_type unaligned(Iter const& iter)
        {
            return boost::simd::load<vector_pack_type>(std::addressof(*iter));
        }
    };

    template <typename V, typename T, std::size_t N, typename Abi>
    struct vector_pack_load<V, boost::simd::pack<T, N, Abi> >
    {
        typedef typename rebind_pack<V, boost::simd::pack<T, N, Abi> >::type
            value_type;

        template <typename Iter>
        static value_type aligned(Iter const& iter)
        {
            return *iter;
        }

        template <typename Iter>
        static value_type unaligned(Iter const& iter)
        {
            return *iter;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename V, typename Value_type, typename Enable>
    struct vector_pack_store
    {
        template <typename Iter>
        static void aligned(V const& value, Iter const& iter)
        {
            boost::simd::aligned_store(value, std::addressof(*iter));
        }

        template <typename Iter>
        static void unaligned(V const& value, Iter const& iter)
        {
            boost::simd::store(value, std::addressof(*iter));
        }
    };

    template <typename V, typename T, std::size_t N, typename Abi>
    struct vector_pack_store<V, boost::simd::pack<T, N, Abi> >
    {
        template <typename Iter>
        static void aligned(V const& value, Iter const& iter)
        {
            *iter = value;
        }

        template <typename Iter>
        static void unaligned(V const& value, Iter const& iter)
        {
            *iter = value;
        }
    };
}}}

#endif
#endif
