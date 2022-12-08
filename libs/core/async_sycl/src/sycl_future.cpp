//  Copyright (c) 2022 Gregor Daiß
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/allocator_support/internal_allocator.hpp>
#include <hpx/assert.hpp>
#include <hpx/config.hpp>
#include <hpx/async_sycl/sycl_future.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)


namespace hpx { namespace sycl { namespace experimental { namespace detail {

    hpx::future<void> get_future(cl::sycl::event command_event)
    {
        return get_future(hpx::util::internal_allocator<>{}, command_event);
    }
}}}}    // namespace hpx::cuda::experimental::detail
#endif
