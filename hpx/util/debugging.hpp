//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2017      Denis Blank
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DEBUGGING_HPP
#define HPX_UTIL_DEBUGGING_HPP

#include <hpx/config.hpp>
#include <string>

namespace hpx {
namespace util {
    /// Attaches a debugger if \c category is equal to the configuration entry
    /// hpx.attach-debugger.
    void HPX_EXPORT may_attach_debugger(std::string const& category);
}    // end namespace util
}    // end namespace hpx

#endif    // HPX_UTIL_DEBUGGING_HPP
