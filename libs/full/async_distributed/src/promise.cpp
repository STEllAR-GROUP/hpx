//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/async_distributed/detail/promise_lco.hpp>
#include <hpx/components_base/component_type.hpp>
#include <hpx/thread_support/atomic_count.hpp>

namespace hpx { namespace traits { namespace detail {

    util::atomic_count unique_type(
        static_cast<long>(components::component_last));
}}}    // namespace hpx::traits::detail

using promise_lco_void = hpx::components::managed_component<
    hpx::lcos::detail::promise_lco<void, hpx::util::unused_type>>;

HPX_REGISTER_COMPONENT_HEAP(promise_lco_void);
