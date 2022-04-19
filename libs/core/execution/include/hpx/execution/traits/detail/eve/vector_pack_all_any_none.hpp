//  Copyright (c) 2022 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR_EVE)
#include <cstddef>

#include <eve/function/all.hpp>
#include <eve/function/any.hpp>
#include <eve/function/none.hpp>

namespace hpx { namespace parallel { namespace traits {
    ///////////////////////////////////////////////////////////////////////
    template <typename Mask>
    HPX_HOST_DEVICE HPX_FORCEINLINE std::size_t all_of(Mask const& msk)
    {
        return eve::all(msk);
    }

    ///////////////////////////////////////////////////////////////////////
    template <typename Mask>
    HPX_HOST_DEVICE HPX_FORCEINLINE std::size_t any_of(Mask const& msk)
    {
        return eve::any(msk);
    }

    ///////////////////////////////////////////////////////////////////////
    template <typename Mask>
    HPX_HOST_DEVICE HPX_FORCEINLINE std::size_t none_of(Mask const& msk)
    {
        return eve::none(msk);
    }
}}}    // namespace hpx::parallel::traits

#endif
