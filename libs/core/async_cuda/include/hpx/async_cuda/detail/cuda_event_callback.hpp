//  Copyright (c) 2021 ETH Zurich
//  Copyright (c) 2020 John Biddiscombe
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file provides functionality similar to CUDA's built-in
// cudaStreamAddCallback, with the difference that an event is recorded and an
// HPX scheduler polls for the completion of the event. When the event is ready,
// a callback is called.

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_cuda/custom_gpu_api.hpp>
#include <hpx/functional/move_only_function.hpp>
#include <hpx/threading_base/thread_pool_base.hpp>

#include <string>

namespace hpx { namespace cuda { namespace experimental { namespace detail {
    using event_callback_function_type =
        hpx::move_only_function<void(cudaError_t)>;

    HPX_CORE_EXPORT void add_event_callback(
        event_callback_function_type&& f, cudaStream_t stream, int device = 0);

    HPX_CORE_EXPORT void register_polling(hpx::threads::thread_pool_base& pool);
    HPX_CORE_EXPORT void unregister_polling(
        hpx::threads::thread_pool_base& pool);
}}}}    // namespace hpx::cuda::experimental::detail
