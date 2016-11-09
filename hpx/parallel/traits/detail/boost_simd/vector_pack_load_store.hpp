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
        typedef boost::simd::pack<NewT, N> type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename V, typename Enable>
    struct vector_pack_load
    {
        template <typename Iter>
        static typename rebind_pack<
            V, typename std::iterator_traits<Iter>::value_type
        >::type
        aligned(Iter const& iter)
        {
            typedef typename rebind_pack<
                    V, typename std::iterator_traits<Iter>::value_type
                >::type vector_pack_type;

            return boost::simd::aligned_load<vector_pack_type>(
                std::addressof(*iter));
        }

        template <typename Iter>
        static typename rebind_pack<
            V, typename std::iterator_traits<Iter>::value_type
        >::type
        unaligned(Iter const& iter)
        {
            typedef typename rebind_pack<
                    V, typename std::iterator_traits<Iter>::value_type
                >::type vector_pack_type;

            return boost::simd::load<vector_pack_type>(std::addressof(*iter));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename V, typename Enable>
    struct vector_pack_store
    {
        template <typename Iter_>
        static void aligned(V const& value, Iter_ const& iter)
        {
            boost::simd::aligned_store(value, std::addressof(*iter));
        }

        template <typename Iter_>
        static void unaligned(V const& value, Iter_ const& iter)
        {
            boost::simd::store(value, std::addressof(*iter));
        }
    };
}}}

#endif
#endif
