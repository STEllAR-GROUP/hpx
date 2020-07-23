//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/synchronization/barrier.hpp>

namespace hpx {

    template <typename OnCompletion = lcos::local::detail::empty_oncompletion>
    using barrier = lcos::local::cpp20_barrier<OnCompletion>;
}
