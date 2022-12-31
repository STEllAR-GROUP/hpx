//  Copyright (c) 2022 Srinivas Yadav
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
    template <typename T, typename Abi, typename Reduce>
    HPX_HOST_DEVICE HPX_FORCEINLINE T reduce(
        Reduce r, Vc::Vector<T, Abi> const& val) noexcept
    {
        T init = val[0];
        for (std::size_t i = 1; i != val.size(); i++)
        {
            init = HPX_INVOKE(r, init, val[i]);
        }
        return init;
    }
}    // namespace hpx::parallel::traits

#endif
