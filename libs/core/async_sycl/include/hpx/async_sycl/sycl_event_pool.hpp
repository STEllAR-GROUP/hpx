//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once


#include <boost/lockfree/stack.hpp>

#include <CL/sycl.hpp> 

// useless... we get the event from the other thing anyways - so it'll always be created and copied elsewhere
// Focus on the event callbacks instead
namespace hpx { namespace sycl { namespace experimental {
#if defined(__HIPSYCL__)
     using syclEvent_t = cl::sycl::event;
#else
     using syclEvent_t = ::sycl::event;
#endif


    // a pool of syclEvent_t objects.
    // Since allocation of a sycl event passes into the sycl runtime
    // it might be an expensive operation, so we pre-allocate a pool
    // of them at startup.
    struct sycl_event_pool
    {
        static constexpr int initial_events_in_pool = 128;

        static sycl_event_pool& get_event_pool()
        {
            static sycl_event_pool event_pool_;
            return event_pool_;
        }

        // create a bunch of events on initialization
        sycl_event_pool()
          : free_list_(initial_events_in_pool)
        {
            for (int i = 0; i < initial_events_in_pool; ++i)
            {
                add_event_to_pool();
            }
        }

        // on destruction, all objects in stack will be freed
        ~sycl_event_pool()
        {
            syclEvent_t event;
            bool ok = true;
            while (ok)
            {
                ok = free_list_.pop(event);
                /* if (ok) */
                /*     check_sycl_error(syclEventDestroy(event)); */
            }
        }

        inline bool pop(syclEvent_t &event)
        {
            // pop an event off the pool, if that fails, create a new one
            while (!free_list_.pop(event))
            {
                add_event_to_pool();
            }
            return true;
        }

        inline bool push(syclEvent_t event)
        {
            return free_list_.push(event);
        }

    private:
        void add_event_to_pool()
        {
            syclEvent_t event;
            // Create an sycl_event to query a CUDA/CUBLAS kernel for completion.
            // Timing is disabled for performance. [1]
            //
            // [1]: CUDA Runtime API, section 5.5 sycl_event Management
            /* check_sycl_error( */
            /*     syclEventCreateWithFlags(&event, syclEventDisableTiming)); */
            free_list_.push(event);
        }

        // pool is dynamically sized and can grow if needed
        boost::lockfree::stack<syclEvent_t, boost::lockfree::fixed_sized<false>>
            free_list_;
    };
}}}    // namespace hpx::sycl_integration::experimental
