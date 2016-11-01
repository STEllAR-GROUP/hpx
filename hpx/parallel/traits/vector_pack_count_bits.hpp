//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TRAITS_VECTOR_PACK_COUNT_BITS_OCT_31_2016_0649PM)
#define HPX_PARALLEL_TRAITS_VECTOR_PACK_COUNT_BITS_OCT_31_2016_0649PM

#include <hpx/config.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace traits
{
    HPX_HOST_DEVICE HPX_FORCEINLINE
    std::size_t count_bits(bool value)
    {
        return value ? 1 : 0;
    }
}}}

#if defined(HPX_HAVE_DATAPAR)

#include <hpx/parallel/traits/detail/vc/vector_pack_count_bits.hpp>
#include <hpx/parallel/traits/detail/boost_simd/vector_pack_count_bits.hpp>

#endif
#endif

