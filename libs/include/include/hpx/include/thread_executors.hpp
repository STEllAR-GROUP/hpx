//  Copyright (c) 2007-2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY)
#include <hpx/runtime_local/pool_executor.hpp>
#include <hpx/runtime_local/service_executors.hpp>
#include <hpx/thread_executors/current_executor.hpp>
#include <hpx/thread_executors/default_executor.hpp>
#include <hpx/thread_executors/embedded_thread_pool_executors.hpp>
#include <hpx/thread_executors/thread_executor.hpp>
#include <hpx/thread_executors/thread_pool_os_executors.hpp>
#include <hpx/thread_executors/thread_timed_execution.hpp>
#endif
