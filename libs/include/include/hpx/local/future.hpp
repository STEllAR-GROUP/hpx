//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/modules/async_base.hpp>
#include <hpx/modules/async_combinators.hpp>
#include <hpx/modules/async_local.hpp>
#include <hpx/modules/execution.hpp>
#include <hpx/modules/futures.hpp>
#include <hpx/modules/lcos_local.hpp>

namespace hpx {
    using hpx::lcos::local::packaged_task;
}
