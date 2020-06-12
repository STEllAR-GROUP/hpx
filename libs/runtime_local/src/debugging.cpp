//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2017      Denis Blank
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/debugging/attach_debugger.hpp>
#include <hpx/runtime_local/config_entry.hpp>
#include <hpx/runtime_local/debugging.hpp>

#include <string>

namespace hpx { namespace util {
    void may_attach_debugger(std::string const& category)
    {
        if (get_config_entry("hpx.attach_debugger", "") == category)
        {
            attach_debugger();
        }
    }
}}    // namespace hpx::util
