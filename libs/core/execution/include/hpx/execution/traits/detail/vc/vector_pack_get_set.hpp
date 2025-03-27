//  Copyright (c) 2022 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR_VC)
#include <hpx/concepts/concepts.hpp>
#include <hpx/execution/traits/vector_pack_alignment_size.hpp>

#include <cstddef>

namespace hpx::parallel::traits {

    ///////////////////////////////////////////////////////////////////////
    template <typename Vector,
        HPX_CONCEPT_REQUIRES_(
            is_vector_pack_v<Vector> || is_scalar_vector_pack_v<Vector>)>
    HPX_HOST_DEVICE HPX_FORCEINLINE auto get(
        Vector& vec, std::size_t index) noexcept
    {
        return vec[index];
    }

    ///////////////////////////////////////////////////////////////////////
    template <typename Vector, typename T,
        HPX_CONCEPT_REQUIRES_(
            is_vector_pack_v<Vector> || is_scalar_vector_pack_v<Vector>)>
    HPX_HOST_DEVICE HPX_FORCEINLINE auto set(
        Vector& vec, std::size_t index, T val) noexcept
    {
        vec[index] = val;
    }
}    // namespace hpx::parallel::traits

#endif
