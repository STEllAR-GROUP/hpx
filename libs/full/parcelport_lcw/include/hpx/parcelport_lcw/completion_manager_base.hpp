//  Copyright (c) 2025 Jiakun Yan
//  Copyright (c) 2014-2023 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCW)

#include <hpx/modules/lcw_base.hpp>

namespace hpx::parcelset::policies::lcw {
    struct completion_manager_base
    {
        virtual ~completion_manager_base() {}
        virtual ::lcw::comp_t alloc_completion() = 0;
        virtual void enqueue_completion(::lcw::comp_t comp) = 0;
        virtual bool poll(::lcw::request_t& request) = 0;
        virtual ::lcw::comp_t get_completion_object()
        {
            return nullptr;
        }
    };
}    // namespace hpx::parcelset::policies::lcw

#endif
