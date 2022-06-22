//  Copyright (c) 2022 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR_VC)
#include <cstddef>

namespace hpx { namespace parallel { namespace traits {
    ///////////////////////////////////////////////////////////////////////
    template <typename Vector>
    HPX_HOST_DEVICE HPX_FORCEINLINE auto get(Vector& vec, std::size_t index)
    {
        return vec[index];
    }

    ///////////////////////////////////////////////////////////////////////
    template <typename Vector, typename T>
    HPX_HOST_DEVICE HPX_FORCEINLINE auto set(
        Vector& vec, std::size_t index, T val)
    {
        vec[index] = val;
    }
}}}    // namespace hpx::parallel::traits

#endif
