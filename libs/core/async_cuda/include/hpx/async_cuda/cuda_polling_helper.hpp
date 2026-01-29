//  Copyright (c) 2021 ETH Zurich
//  Copyright (c) 2020 John Biddiscombe
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_cuda/detail/cuda_event_callback.hpp>
#include <hpx/modules/runtime_local.hpp>
#include <hpx/modules/threading_base.hpp>

#include <string>

namespace hpx::cuda::experimental {

    // -----------------------------------------------------------------
    // This RAII helper class enables polling for a scoped block
    HPX_CXX_CORE_EXPORT struct [[nodiscard]] enable_user_polling
    {
        enable_user_polling()
          : pool_name_()
        {
            // install polling loop on requested thread pool
            detail::register_polling(hpx::resource::get_thread_pool(0));
        }

        explicit enable_user_polling(std::string const& pool_name)
          : pool_name_(pool_name)
        {
            // install polling loop on requested thread pool
            if (pool_name_.empty())
            {
                detail::register_polling(hpx::resource::get_thread_pool(0));
            }
            else
            {
                detail::register_polling(
                    hpx::resource::get_thread_pool(pool_name_));
            }
        }

        ~enable_user_polling()
        {
            if (pool_name_.empty())
            {
                detail::unregister_polling(hpx::resource::get_thread_pool(0));
            }
            else
            {
                detail::unregister_polling(
                    hpx::resource::get_thread_pool(pool_name_));
            }
        }

    private:
        std::string pool_name_;
    };
}    // namespace hpx::cuda::experimental
