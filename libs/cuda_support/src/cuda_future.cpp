//  Copyright (c) 2020 John Biddiscombe
//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#include <hpx/allocator_support/internal_allocator.hpp>
#include <hpx/assert.hpp>
#include <hpx/cuda_support/target.hpp>
#include <hpx/errors.hpp>
#include <hpx/futures/traits/future_access.hpp>
#include <hpx/memory/intrusive_ptr.hpp>
#include <hpx/runtime/find_here.hpp>
#include <hpx/runtime/naming/id_type_impl.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/threading_base/thread_helpers.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include <cuda_runtime.h>

namespace hpx { namespace cuda { namespace detail {
    runtime_registration_wrapper::runtime_registration_wrapper(hpx::runtime* rt)
      : rt_(rt)
    {
        HPX_ASSERT(rt);

        // Register this thread with HPX, this should be done once for
        // each external OS-thread intended to invoke HPX functionality.
        // Calling this function more than once on the same thread will
        // report an error.
        hpx::error_code ec(hpx::lightweight);    // ignore errors
        hpx::register_thread(rt_, "cuda", ec);
    }

    runtime_registration_wrapper::~runtime_registration_wrapper()
    {
        // Unregister the thread from HPX, this should be done once in
        // the end before the external thread exists.
        hpx::unregister_thread(rt_);
    }

    // -------------------------------------------------------------
    // main API call to get a future from a stream
    hpx::future<void> get_future(cudaStream_t stream)
    {
        return get_future(hpx::util::internal_allocator<>{}, stream);
    }

}}}    // namespace hpx::cuda::detail
