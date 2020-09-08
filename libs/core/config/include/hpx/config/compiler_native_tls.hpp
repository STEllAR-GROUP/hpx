//  Copyright (c) 2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nodeprecated_macros

#pragma once

#include <hpx/config/defines.hpp>

/// This macro is replaced with the compiler specific keyword attribute to mark
/// a variable as thread local. For more details see
/// `<https://en.cppreference.com/w/cpp/keyword/thread_local`__.
///
/// This macro is deprecated. It is always replaced with the ``thread_local``
/// keyword. Prefer using ``thread_local`` directly instead.
#define HPX_NATIVE_TLS thread_local
