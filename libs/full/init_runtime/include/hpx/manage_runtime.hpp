//  Copyright (c)      2025 Agustin Berge
//  Copyright (c) 2016-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file manage_runtime.hpp

#pragma once

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
        int start(
            int argc, char** argv, const init_params& init_args = init_params())
        {
            HPX_ASSERT(!running_);

            function<int(int, char**)> start_function =
                bind_front(&manage_runtime::hpx_main, this);

            const int ret = hpx::start(start_function, argc, argv, init_args);
            if (!ret)
                return ret;

            running_ = true;

            // wait for the main HPX thread (hpx_main below) to have started running
            std::unique_lock<std::mutex> lk(startup_mtx_);
            startup_cond_.wait(lk, [&] { return rts_ != nullptr; });

            return ret;
        }

        int stop()
        {
            HPX_ASSERT(running_);

            running_ = false;

            // signal to `hpx_main` below to tear down the runtime
            {
                std::lock_guard<spinlock> lk(stop_mtx_);
                rts_ = nullptr;
            }
            stop_cond_.notify_one();

            // wait for the runtime to exit
            return hpx::stop();
        }

        runtime* get_runtime_ptr() const noexcept
        {
            return rts_;
        }

    private:
        // Main HPX thread, does nothing but wait for the application to exit
        int hpx_main(int, char*[])
        {
            // signal to `start` that thread has started running.
            {
                std::lock_guard<std::mutex> lk(startup_mtx_);
                rts_ = hpx::get_runtime_ptr();
            }
            startup_cond_.notify_one();

            // wait for `stop` to be called.
            {
                std::unique_lock<spinlock> lk(stop_mtx_);
                stop_cond_.wait(lk, [&] { return rts_ == nullptr; });
            }

            // tell the runtime it's ok to exit
            return hpx::finalize();
        }

    private:
        bool running_ = false;
        runtime* rts_ = nullptr;

        std::mutex startup_mtx_;
        std::condition_variable startup_cond_;

        spinlock stop_mtx_;
        condition_variable_any stop_cond_;
    };
}    // namespace hpx
