//  Copyright (c) 2020 John Biddiscombe
//  Copyright (c) 2020 Teodor Nikolov
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <deque>

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

        static cuda_event_pool& get_event_pool()
        {
            static cuda_event_pool event_pool_;
            return event_pool_;
        }

        // create a bunch of events on initialization
        cuda_event_pool()
          : max_number_devices_(0)
        {
            check_cuda_error(cudaGetDeviceCount(&max_number_devices_));
            HPX_ASSERT_MSG(max_number_devices_ > 0,
                "CUDA polling enabled and called, yet no CUDA device found!");
            for (int device = 0; device < max_number_devices_; device++) {
              check_cuda_error(cudaSetDevice(device));
              free_lists_.emplace_back(initial_events_in_pool);
              for (int i = 0; i < initial_events_in_pool; ++i)
              {
                  add_event_to_pool(device);
              }
            }
        }

        // on destruction, all objects in stack will be freed
        ~cuda_event_pool()
        {
            HPX_ASSERT_MSG(free_lists_.size != max_number_devices_,
                "Number of CUDA event pools does not match the number of devices!");
            for (int device = 0; device < max_number_devices_; device++) {
              check_cuda_error(cudaSetDevice(device));
              cudaEvent_t event;
              bool ok = true;
              while (ok)
              {
                  ok = free_lists_[device].pop(event);
                  if (ok)
                      check_cuda_error(cudaEventDestroy(event));
              }
            }
        }

        inline bool pop(cudaEvent_t& event, int device)
        {
            HPX_ASSERT_MSG(device > 0 && device < max_number_devices_,
                "Accessing CUDA event pool with invalid device ID!");
            // pop an event off the pool, if that fails, create a new one
            while (!free_lists_[device].pop(event))
            {
                add_event_to_pool(device);
            }
            return true;
        }

        inline bool push(cudaEvent_t event, int device)
        {
            HPX_ASSERT_MSG(device > 0 && device < max_number_devices_,
                "Accessing CUDA event pool with invalid device ID!");
            return free_lists_[device].push(event);
        }

    private:
        void add_event_to_pool(int device)
        {
            check_cuda_error(cudaSetDevice(device));
            cudaEvent_t event;
            // Create an cuda_event to query a CUDA/CUBLAS kernel for completion.
            // Timing is disabled for performance. [1]
            //
            // [1]: CUDA Runtime API, section 5.5 cuda_event Management
            check_cuda_error(
                cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
            free_lists_[device].push(event);
        }
        int max_number_devices_;

        // One pool per GPU - each pool is dynamically sized and can grow if needed
        std::deque<hpx::lockfree::stack<cudaEvent_t>> free_lists_;
    };
}}}    // namespace hpx::cuda::experimental
