//  Copyright (c) 2020-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_THREAD_DESCRIPTION)

#include <hpx/modules/errors.hpp>
#include <hpx/threading_base/thread_description.hpp>
#include <hpx/threading_base/threading_base_fwd.hpp>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx::threads::detail {

    struct HPX_CORE_EXPORT reset_lco_description
    {
        reset_lco_description(threads::thread_id_type const& id,
            threads::thread_description const& description,
            error_code& ec = throws);

        ~reset_lco_description();

        threads::thread_id_type id_;
        threads::thread_description old_desc_;
        error_code& ec_;
    };
}    // namespace hpx::threads::detail

#include <hpx/config/warnings_suffix.hpp>

#endif    // HPX_HAVE_THREAD_DESCRIPTION
