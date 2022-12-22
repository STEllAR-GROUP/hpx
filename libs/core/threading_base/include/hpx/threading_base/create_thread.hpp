//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/threading_base/threading_base_fwd.hpp>

namespace hpx::threads::detail {

    HPX_CORE_EXPORT void create_thread(policies::scheduler_base* scheduler,
        threads::thread_init_data& data, threads::thread_id_ref_type& id,
        error_code& ec = throws);
}    // namespace hpx::threads::detail
