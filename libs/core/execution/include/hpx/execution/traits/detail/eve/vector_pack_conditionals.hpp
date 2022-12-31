//  Copyright (c) 2022 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR_EVE)
#include <eve/eve.hpp>

namespace hpx::parallel::traits {

    ////////////////////////////////////////////////////////////////////
    template <typename T, typename Abi>
    HPX_HOST_DEVICE HPX_FORCEINLINE eve::wide<T, Abi> choose(
        eve::logical<eve::wide<T, Abi>> const& msk,
        eve::wide<T, Abi> const& v_true,
        eve::wide<T, Abi> const& v_false) noexcept
    {
        return eve::if_else(msk, v_true, v_false);
    }

    ////////////////////////////////////////////////////////////////////
    template <typename T, typename Abi>
    HPX_HOST_DEVICE HPX_FORCEINLINE void mask_assign(
        eve::logical<eve::wide<T, Abi>> const& msk, eve::wide<T, Abi>& v,
        eve::wide<T, Abi> const& val) noexcept
    {
        v = eve::if_else(msk, val, v);
    }
}    // namespace hpx::parallel::traits

#endif
