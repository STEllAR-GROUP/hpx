//  Copyright (c) 2019-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/threading_base/thread_data.hpp>

namespace hpx { namespace threads { namespace detail {

    HPX_CORE_EXPORT bool execute_thread(thread_id_ref_type const& thrd);

}}}    // namespace hpx::threads::detail
