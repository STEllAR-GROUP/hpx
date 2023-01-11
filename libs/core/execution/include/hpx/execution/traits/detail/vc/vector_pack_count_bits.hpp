//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c) 2016 Matthias Kretz
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR_VC)
#include <cstddef>

#include <Vc/Vc>
#include <Vc/global.h>

namespace hpx::parallel::traits {

    ///////////////////////////////////////////////////////////////////////
    template <typename T, typename Abi>
    HPX_HOST_DEVICE HPX_FORCEINLINE std::size_t count_bits(
        Vc::Mask<T, Abi> const& mask) noexcept
    {
        return mask.count();
    }
}    // namespace hpx::parallel::traits

#endif
