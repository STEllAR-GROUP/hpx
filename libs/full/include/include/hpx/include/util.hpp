//  Copyright (c) 2007-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/command_line_handling/parse_command_line.hpp>
#include <hpx/execution_base/this_thread.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/modules/functional.hpp>
#include <hpx/modules/iterator_support.hpp>
#include <hpx/modules/preprocessor.hpp>
#include <hpx/modules/string_util.hpp>
#include <hpx/modules/thread_support.hpp>
#include <hpx/modules/type_support.hpp>
#include <hpx/modules/util.hpp>
#include <hpx/runtime_local/interval_timer.hpp>
#include <hpx/threading_base/annotated_function.hpp>
#include <hpx/timing/high_resolution_clock.hpp>
#include <hpx/timing/high_resolution_timer.hpp>
#include <hpx/unwrap.hpp>

#include <hpx/parallel/segmented_algorithms/traits/zip_iterator.hpp>
