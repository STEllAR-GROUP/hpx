//  Copyright (c) 2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/reference_wrapper.hpp>
#include <hpx/modules/serialization.hpp>

namespace hpx::serialization {

    HPX_CXX_CORE_EXPORT template <typename T>
        requires(traits::needs_reference_semantics_v<T>)
    void serialize(input_archive& ar, hpx::reference_wrapper<T>& ref, unsigned)
    {
        T val;
        ar >> val;
        ref = hpx::ref(HPX_MOVE(val));
    }

    HPX_CXX_CORE_EXPORT template <typename T>
        requires(traits::needs_reference_semantics_v<T>)
    void serialize(output_archive& ar, hpx::reference_wrapper<T>& ref, unsigned)
    {
        ar << ref.get();
    }
}    // namespace hpx::serialization
