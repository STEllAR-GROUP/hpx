//  Copyright (c) 2007-2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/type_support/static_reinit_interface.hpp>

namespace hpx::util {

    ///////////////////////////////////////////////////////////////////////////
    // initialize static_reinit interface function wrappers
    HPX_CORE_EXPORT struct static_reinit_interface_functions&
    static_reinit_init();
}    // namespace hpx::util
