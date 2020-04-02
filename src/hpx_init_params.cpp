//  Copyright (c)      2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init_params.hpp>
#include <hpx/runtime/shutdown_function.hpp>
#include <hpx/runtime/startup_function.hpp>
#include <hpx/type_support/unused.hpp>


namespace hpx { namespace detail {
    HPX_MAYBE_UNUSED int dummy_argc = 1;
    // TODO: make it only one parameter, probably add a cast
    char *default_argv[2] = { app_name, nullptr };
    HPX_MAYBE_UNUSED char **dummy_argv = default_argv;
}}  // namespace hpx::detail
