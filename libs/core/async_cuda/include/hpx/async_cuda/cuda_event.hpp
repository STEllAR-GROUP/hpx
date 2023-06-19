//  Copyright (c) 2020 John Biddiscombe
//  Copyright (c) 2020 Teodor Nikolov
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/async_cuda/cuda_exception.hpp>
#include <hpx/async_cuda/custom_gpu_api.hpp>
#include <hpx/concurrency/stack.hpp>

namespace hpx { namespace cuda { namespace experimental {

    // a pool of cudaEvent_t objects.
    // Since allocation of a cuda event passes into the cuda runtime
    // it might be an expensive operation, so we pre-allocate a pool
    // of them at startup.
    struct cuda_event_pool
    {
        static constexpr int initial_events_in_pool = 128;

        const int device_id;

        static cuda_event_pool& get_event_pool(size_t device_id)
        {
            static std::array<cuda_event_pool, 4> event_pool_{0, 1, 2, 3};
            return event_pool_[device_id];
        }

        // create a bunch of events on initialization
        cuda_event_pool(int device_id)
          : device_id(device_id), free_list_(initial_events_in_pool)
        {
            check_cuda_error(cudaSetDevice(device_id));
            for (int i = 0; i < initial_events_in_pool; ++i)
            {
                add_event_to_pool();
            }
            std::cerr << "Created " << device_id << std::endl;
        }

        // on destruction, all objects in stack will be freed
        ~cuda_event_pool()
        {
            check_cuda_error(cudaSetDevice(device_id));
            cudaEvent_t event;
            bool ok = true;
            while (ok)
            {
                ok = free_list_.pop(event);
                if (ok)
                    check_cuda_error(cudaEventDestroy(event));
            }
        }

        inline bool pop(cudaEvent_t& event)
        {
            // pop an event off the pool, if that fails, create a new one
            while (!free_list_.pop(event))
            {
                add_event_to_pool();
            }
            return true;
        }

        inline bool push(cudaEvent_t event)
        {
            return free_list_.push(event);
        }

    private:
        void add_event_to_pool()
        {
            check_cuda_error(cudaSetDevice(device_id));
            cudaEvent_t event;
            // Create an cuda_event to query a CUDA/CUBLAS kernel for completion.
            // Timing is disabled for performance. [1]
            //
            // [1]: CUDA Runtime API, section 5.5 cuda_event Management
            check_cuda_error(
                cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
            free_list_.push(event);
        }

        // pool is dynamically sized and can grow if needed
        hpx::lockfree::stack<cudaEvent_t> free_list_;
    };
}}}    // namespace hpx::cuda::experimental
