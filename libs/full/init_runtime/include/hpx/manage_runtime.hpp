//  Copyright (c) 2016-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/condition_variable.hpp>
#include <hpx/functional.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/runtime_local.hpp>
#include <hpx/mutex.hpp>
#include <hpx/thread.hpp>

#include <cstdlib>
#include <mutex>

namespace hpx {

    class manage_runtime
    {
    public:
        manage_runtime(
            int argc, char** argv, const init_params& init_args = init_params())
          : running_(false)
          , rts_(nullptr)
        {
#if defined(HPX_WINDOWS)
            detail::init_winsocket();
#endif

            function<int(int, char**)> start_function =
                bind_front(&manage_runtime::hpx_main, this);

            if (!start(start_function, argc, argv, init_args))
            {
                // Something went wrong while initializing the runtime.
                // This early we can't generate any output, just bail out.
                std::abort();
            }

            // Wait for the main HPX thread (hpx_main below) to have started running
            std::unique_lock<std::mutex> lk(startup_mtx_);
            while (!running_)
                startup_cond_.wait(lk);
        }

        ~manage_runtime()
        {
            // notify hpx_main above to tear down the runtime
            {
                std::lock_guard<spinlock> lk(mtx_);
                rts_ = nullptr;    // reset pointer
            }

            cond_.notify_one();    // signal exit

            // wait for the runtime to exit
            stop();
        }

        runtime* get_runtime_ptr() const noexcept
        {
            return rts_;
        }

    protected:
        // Main HPX thread, does nothing but wait for the application to exit
        int hpx_main(int, char*[])
        {
            // Store a pointer to the runtime here.
            rts_ = hpx::get_runtime_ptr();

            // Signal to constructor that thread has started running.
            {
                std::lock_guard<std::mutex> lk(startup_mtx_);
                running_ = true;
            }

            startup_cond_.notify_one();

            // Here other HPX specific functionality could be invoked...

            // Now, wait for destructor to be called.
            {
                std::unique_lock<spinlock> lk(mtx_);
                if (rts_ != nullptr)
                    cond_.wait(lk);
            }

            // tell the runtime it's ok to exit
            return finalize();
        }

    private:
        spinlock mtx_;
        condition_variable_any cond_;

        std::mutex startup_mtx_;
        std::condition_variable startup_cond_;
        bool running_;

        runtime* rts_;
    };
}    // namespace hpx
