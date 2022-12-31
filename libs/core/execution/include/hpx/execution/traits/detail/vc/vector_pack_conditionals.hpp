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

    ////////////////////////////////////////////////////////////////////
    template <typename T, typename Abi>
    HPX_HOST_DEVICE HPX_FORCEINLINE auto choose(Vc::Mask<T, Abi> const& msk,
        Vc::Vector<T, Abi> const& v_true,
        Vc::Vector<T, Abi> const& v_false) noexcept
    {
        Vc::Vector<T, Abi> v;
        where(msk, v) = v_true;
        where(!msk, v) = v_false;
        return v;
    }

    ////////////////////////////////////////////////////////////////////
    template <typename T, typename Abi>
    HPX_HOST_DEVICE HPX_FORCEINLINE void mask_assign(
        Vc::Mask<T, Abi> const& msk, Vc::Vector<T, Abi>& v,
        Vc::Vector<T, Abi> const& val) noexcept
    {
        where(msk, v) = val;
    }
}    // namespace hpx::parallel::traits

#endif
