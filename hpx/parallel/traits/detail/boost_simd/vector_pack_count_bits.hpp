//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_DATAPAR_BOOST_SIMD_COUNT_BITS_SEP_22_2016_0220PM)
#define HPX_PARALLEL_DATAPAR_BOOST_SIMD_COUNT_BITS_SEP_22_2016_0220PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR_BOOST_SIMD)
#include <cstddef>

#include <boost/simd.hpp>
#include <boost/simd/function/sum.hpp>

namespace hpx { namespace parallel { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T, std::size_t N, typename Abi>
    HPX_HOST_DEVICE HPX_FORCEINLINE std::size_t
    count_bits(boost::simd::pack<boost::simd::logical<T>, N, Abi> const& mask)
    {
        return boost::simd::sum(mask);
    }
}}}

#endif
#endif

