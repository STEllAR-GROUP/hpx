//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c) 2016 Matthias Kretz
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_DATAPAR_VC_COUNT_BITS_SEP_07_2016_1217PM)
#define HPX_PARALLEL_DATAPAR_VC_COUNT_BITS_SEP_07_2016_1217PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR_VC)
#include <cstddef>

#include <Vc/version.h>

#if Vc_IS_VERSION_1

#include <Vc/Vc>

namespace hpx { namespace parallel { namespace traits
{
    ///////////////////////////////////////////////////////////////////////
    template <typename T, typename Abi>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    std::size_t count_bits(Vc::Mask<T, Abi> const& mask)
    {
        return mask.count();
    }
}}}

#else

#include <Vc/datapar>

namespace hpx { namespace parallel { namespace traits
{
    ///////////////////////////////////////////////////////////////////////
    template <typename T, typename Abi>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    std::size_t count_bits(Vc::mask<T, Abi> const& mask)
    {
        return Vc::popcount(mask);
    }
}}}

#endif  // Vc_IS_VERSION_1

#endif
#endif

