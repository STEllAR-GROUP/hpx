//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TRAITS_VECTOR_PACK_TYPE_BOOST_SIMD_OCT_31_2016_1229PM)
#define HPX_PARALLEL_TRAITS_VECTOR_PACK_TYPE_BOOST_SIMD_OCT_31_2016_1229PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR_BOOST_SIMD)

#include <cstddef>
#include <type_traits>

#include <boost/simd.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename T, std::size_t N, typename Abi>
        struct vector_pack_type
        {
            typedef typename std::conditional<
                    std::is_void<Abi>::value, boost::simd::abi_of_t<T, N>, Abi
                >::type abi_type;

            typedef boost::simd::pack<T, N, abi_type> type;
        };

        template <typename T, typename Abi>
        struct vector_pack_type<T, 0, Abi>
        {
            static std::size_t const N = boost::simd::native_cardinal<T>::value;
            typedef typename std::conditional<
                    std::is_void<Abi>::value, boost::simd::abi_of_t<T, N>, Abi
                >::type abi_type;

            typedef boost::simd::pack<T, N, abi_type> type;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    // Avoid premature instantiation of boost::simd::native_cardinal and
    // boost::simd::abi_of_t.
    template <typename T, std::size_t N, typename Abi>
    struct vector_pack_type
      : detail::vector_pack_type<T, N, Abi>
    {};
}}}

#endif
#endif

