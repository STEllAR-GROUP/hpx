//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Matthias Kretz
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TRAITS_VECTOR_PACK_TYPE_VC_OCT_31_2016_1229PM)
#define HPX_PARALLEL_TRAITS_VECTOR_PACK_TYPE_VC_OCT_31_2016_1229PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR_VC)

#include <cstddef>
#include <type_traits>

#include <Vc/Vc>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename T, std::size_t N, typename Abi>
        struct vector_pack_type
        {
            typedef Vc::SimdArray<T, N> type;
        };

        template <typename T, typename Abi>
        struct vector_pack_type<T, 0, Abi>
        {
            typedef typename std::conditional<
                    std::is_void<Abi>::value, Vc::VectorAbi::Best<T>, Abi
                >::type abi_type;

            typedef Vc::Vector<T, abi_type> type;
        };

        template <typename T, typename Abi>
        struct vector_pack_type<T, 1, Abi>
        {
            typedef Vc::Scalar::Vector<T> type;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, std::size_t N, typename Abi>
    struct vector_pack_type
      : detail::vector_pack_type<T, N, Abi>
    {};

    // don't wrap types twice
    template <typename T, std::size_t N, typename Abi1, typename Abi2>
    struct vector_pack_type<Vc::Vector<T, Abi1>, N, Abi2>
    {
        typedef Vc::Vector<T, Abi1> type;
    };

    template <typename T, std::size_t N1, typename V, std::size_t W,
         std::size_t N2, typename Abi>
    struct vector_pack_type<Vc::SimdArray<T, N1, V, W>, N2, Abi>
    {
        typedef Vc::SimdArray<T, N1, V, W> type;
    };

    template <typename T, std::size_t N, typename Abi>
    struct vector_pack_type<Vc::Scalar::Vector<T>, N, Abi>
    {
        typedef Vc::Scalar::Vector<T> type;
    };
}}}

#endif
#endif

