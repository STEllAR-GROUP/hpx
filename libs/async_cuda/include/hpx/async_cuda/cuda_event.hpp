//  Copyright (c) 2020 John Biddiscombe
//  Copyright (c) 2020 Teodor Nikolov
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/async_cuda/cuda_exception.hpp>
//
#include <boost/lockfree/stack.hpp>
//
#include <cuda_runtime.h>

namespace hpx { namespace cuda { namespace experimental {

    // a pool of cudaEvent_t objects.
    // Since allocation of a cuda event passes into the cuda runtime
    // it might be an expensive operation, so we pre-allocate a pool
    // of them at startup.
    // For now - Assume a maximum of 64 outstanding events is enough
    struct cuda_event_pool
    {
        static constexpr int max_events_in_pool = 64;

        static cuda_event_pool& get_event_pool()
        {
            static cuda_event_pool event_pool_;
            return event_pool_;
        }

        // create a bunch of events on initialization
        cuda_event_pool()
        {
            for (int i = 0; i < max_events_in_pool; ++i)
            {
                cudaEvent_t event;
                // Create an cuda_event to query a CUDA/CUBLAS kernel for completion.
                // Timing is disabled for performance. [1]
                //
                // [1]: CUDA Runtime API, section 5.5 cuda_event Management
                check_cuda_error(
                    cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
                free_list_.push(event);
            }
        }

        // on destruction, all objects in stack will be freed
        ~cuda_event_pool()
        {
            cudaEvent_t event;
            for (int i = 0; i < max_events_in_pool; ++i)
            {
                free_list_.pop(event);
                check_cuda_error(cudaEventDestroy(event));
            }
        }

        inline bool pop(cudaEvent_t& event)
        {
            return free_list_.pop(event);
        }

        inline bool push(cudaEvent_t event)
        {
            return free_list_.push(event);
        }

    private:
        // using a fixed capacity stack means no allocations are
        // needed , throws an exception if the capacity is exceeded
        boost::lockfree::stack<cudaEvent_t,
            boost::lockfree::capacity<max_events_in_pool>>
            free_list_;
    };
}}}    // namespace hpx::cuda::experimental
