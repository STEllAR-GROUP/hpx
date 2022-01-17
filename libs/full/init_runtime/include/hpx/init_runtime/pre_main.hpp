//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)

#include <hpx/runtime_configuration/runtime_mode.hpp>

namespace hpx { namespace detail {

    HPX_EXPORT int pre_main(runtime_mode);
    HPX_EXPORT void post_main();
}}    // namespace hpx::detail

#endif
