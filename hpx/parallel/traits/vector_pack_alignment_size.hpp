//  Copyright (c) 2016-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TRAITS_VECTOR_PACK_ALIGNMENT_SEP_29_2016_0122PM)
#define HPX_PARALLEL_TRAITS_VECTOR_PACK_ALIGNMENT_SEP_29_2016_0122PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR)
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/tuple.hpp>

#include <cstddef>
#include <type_traits>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Enable = void>
    struct is_vector_pack : std::false_type {};

    template <typename T, typename Enable = void>
    struct is_scalar_vector_pack;

    template <typename T, typename Enable>
    struct is_scalar_vector_pack : std::false_type {};

    template <typename T, typename Enable = void>
    struct is_non_scalar_vector_pack : std::false_type {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Enable = void>
    struct vector_pack_alignment;

    template <typename ... Vector>
    struct vector_pack_alignment<hpx::util::tuple<Vector...>,
        typename std::enable_if<
            hpx::util::detail::all_of<is_vector_pack<Vector>...>::value
        >::type>
    {
        typedef typename hpx::util::tuple_element<
                0, hpx::util::tuple<Vector...>
            >::type pack_type;

        static std::size_t const value =
            vector_pack_alignment<pack_type>::value;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Enable = void>
    struct vector_pack_size;

    template <typename ... Vector>
    struct vector_pack_size<hpx::util::tuple<Vector...>,
        typename std::enable_if<
            hpx::util::detail::all_of<is_vector_pack<Vector>...>::value
        >::type>
    {
        typedef typename hpx::util::tuple_element<
                0, hpx::util::tuple<Vector...>
            >::type pack_type;

        static std::size_t const value =
            vector_pack_size<pack_type>::value;
    };
}}}

#if !defined(__CUDACC__)
#include <hpx/parallel/traits/detail/vc/vector_pack_alignment_size.hpp>
#include <hpx/parallel/traits/detail/boost_simd/vector_pack_alignment_size.hpp>
#endif

#endif
#endif

