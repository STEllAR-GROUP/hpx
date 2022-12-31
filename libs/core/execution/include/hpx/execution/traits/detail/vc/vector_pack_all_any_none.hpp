//  Copyright (c) 2021 Srinivas Yadav
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
    HPX_HOST_DEVICE HPX_FORCEINLINE std::size_t all_of(
        Vc::Mask<T, Abi> const& msk) noexcept
    {
        return Vc::all_of(msk);
    }

    ///////////////////////////////////////////////////////////////////////
    template <typename T, typename Abi>
    HPX_HOST_DEVICE HPX_FORCEINLINE std::size_t any_of(
        Vc::Mask<T, Abi> const& msk) noexcept
    {
        return Vc::any_of(msk);
    }

    ///////////////////////////////////////////////////////////////////////
    template <typename T, typename Abi>
    HPX_HOST_DEVICE HPX_FORCEINLINE std::size_t none_of(
        Vc::Mask<T, Abi> const& msk) noexcept
    {
        return Vc::none_of(msk);
    }
}    // namespace hpx::parallel::traits

#endif
