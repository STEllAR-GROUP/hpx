//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CXX23_DEPRECATED_STD_ALIGNED_STORAGE)

#include <cstddef>

// C++23 deprecated std::aligned_storage_t
namespace hpx {

    template <std::size_t Size, std::size_t Align = alignof(std::max_align_t)>
    struct alignas(Align) aligned_storage_t
    {
        std::byte data[Size];
    };
}    // namespace hpx

#else

#include <type_traits>

namespace hpx {

    template <std::size_t Size, std::size_t Align = alignof(std::max_align_t)>
    using aligned_storage_t = std::aligned_storage_t<Size, Align>;
}    // namespace hpx

#endif
