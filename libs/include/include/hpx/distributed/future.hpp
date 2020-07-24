//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/async_distributed/async.hpp>
#include <hpx/async_distributed/async_callback.hpp>
#include <hpx/async_distributed/async_continue_callback.hpp>
#include <hpx/async_distributed/detail/async_colocated.hpp>
#include <hpx/async_distributed/detail/async_colocated_callback.hpp>
#include <hpx/lcos/promise.hpp>

namespace hpx { namespace distribtued {
    using hpx::lcos::promise;
}}    // namespace hpx::distribtued
