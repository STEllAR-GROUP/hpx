//  Copyright (c) 2007-2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/type_support/detail/static_reinit_functions.hpp>
#include <hpx/type_support/static_reinit_interface.hpp>

#include <functional>

namespace hpx::util {

    // This is a global API allowing to register functions to be called before
    // the runtime system is about to start and after the runtime system has
    // been terminated. This is used to initialize/reinitialize all singleton
    // instances.
    void reinit_register(std::function<void()> const& construct,
        std::function<void()> const& destruct)
    {
        detail::reinit_register(construct, destruct);
    }

    // Invoke all globally registered construction functions
    void reinit_construct()
    {
        detail::reinit_construct();
    }

    // Invoke all globally registered destruction functions
    void reinit_destruct()
    {
        detail::reinit_destruct();
    }
}    // namespace hpx::util
