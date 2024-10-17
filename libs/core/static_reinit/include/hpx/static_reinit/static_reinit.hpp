//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/function.hpp>

namespace hpx::util {

    ///////////////////////////////////////////////////////////////////////////
    // This is a global API allowing to register functions to be called before
    // the runtime system is about to start and after the runtime system has
    // been terminated. This is used to initialize/reinitialize all
    // singleton instances.
    HPX_CORE_EXPORT void reinit_register(hpx::function<void()> const& construct,
        hpx::function<void()> const& destruct);

    // Invoke all globally registered construction functions
    HPX_CORE_EXPORT void reinit_construct();

    // Invoke all globally registered destruction functions
    HPX_CORE_EXPORT void reinit_destruct();
}    // namespace hpx::util
