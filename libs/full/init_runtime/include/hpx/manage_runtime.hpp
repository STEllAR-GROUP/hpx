//  Copyright (c)      2025 Agustin Berge
//  Copyright (c) 2016-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file manage_runtime.hpp

#pragma once

#include <hpx/config.hpp>

#include <hpx/condition_variable.hpp>
#include <hpx/init.hpp>
#include <hpx/manage_runtime.hpp>
#include <hpx/modules/runtime_local.hpp>
#include <hpx/mutex.hpp>
#include <hpx/thread.hpp>

#include <mutex>

namespace hpx {

    class HPX_EXPORT manage_runtime
    {
    public:
        int start(int argc, char** argv,
            const init_params& init_args = init_params());
        int stop();

        runtime* get_runtime_ptr() const noexcept
        {
            return rts_;
        }

    private:
        // Main HPX thread, does nothing but wait for the application to exit
        int hpx_main(int, char*[]);

    private:
        bool running_ = false;
        runtime* rts_ = nullptr;

        std::mutex startup_mtx_;
        std::condition_variable startup_cond_;

        spinlock stop_mtx_;
        condition_variable_any stop_cond_;
    };
}    // namespace hpx
