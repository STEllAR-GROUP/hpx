//  Copyright (c) 2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <memory>

namespace hpx::serialization {

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Allocator = std::allocator<T>>
    class serialize_buffer;
}    // namespace hpx::serialization
