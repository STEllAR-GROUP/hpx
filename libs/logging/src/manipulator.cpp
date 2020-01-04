// manipulator.cpp

// Boost Logging library
//
// Author: John Torjo, www.torjo.com
//
// Copyright (C) 2007 John Torjo (see www.torjo.com for email)
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org for updates, documentation, and revision history.
// See http://www.torjo.com/log2/ for more details

#include <hpx/logging/manipulator.hpp>

namespace hpx { namespace util { namespace logging {

    namespace formatter {

        manipulator::~manipulator() = default;

    }    // namespace formatter

    namespace destination {

        manipulator::~manipulator() = default;

    }    // namespace destination

}}}    // namespace hpx::util::logging
