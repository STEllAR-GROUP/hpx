//  Copyright (c) 2021 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR_VC)
#include <Vc/Vc>
#include <Vc/global.h>

namespace hpx::parallel::traits {

    ///////////////////////////////////////////////////////////////////////
    template <typename T, typename Abi>
    HPX_HOST_DEVICE HPX_FORCEINLINE int find_first_of(
        Vc::Mask<T, Abi> const& msk) noexcept
    {
        if (Vc::any_of(msk))
        {
            return msk.firstOne();
        }
        return -1;
    }
}    // namespace hpx::parallel::traits

#endif
