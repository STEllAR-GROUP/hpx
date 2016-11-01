//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TRAITS_VECTOR_PACK_ALIGNMENT_SEP_29_2016_0122PM)
#define HPX_PARALLEL_TRAITS_VECTOR_PACK_ALIGNMENT_SEP_29_2016_0122PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR)

#include <type_traits>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace traits
{
    template <typename Iter, typename T, typename Enable = void>
    struct vector_pack_alignment;

    template <typename Iter, typename T, typename Enable = void>
    struct vector_pack_size;

    template <typename T, typename Enable = void>
    struct vector_pack_is_scalar : std::true_type {};
}}}

#include <hpx/parallel/traits/detail/vc/vector_pack_alignment_size.hpp>
#include <hpx/parallel/traits/detail/boost_simd/vector_pack_alignment_size.hpp>

#endif
#endif

