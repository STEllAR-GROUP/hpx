//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>

///////////////////////////////////////////////////////////////////////////////
// This is code specific to Windows
#if defined(HPX_WINDOWS)

#include <cstdint>

///////////////////////////////////////////////////////////////////////////////
// Install the windows performance counters exposing  the HPX counters.
void install_windows_counters();

// Update the data to be exposed as the windows performance counter values.
void update_windows_counters(std::uint64_t value);

// Uninstall the windows performance counters exposing  the HPX counters.
void uninstall_windows_counters();

#endif


#endif
