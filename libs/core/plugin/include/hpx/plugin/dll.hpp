// Copyright (c) 2005-2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/plugin/config.hpp>

#if !defined(HPX_MSVC)
#if !defined(HPX_HAS_DLOPEN)
#define HPX_HAS_DLOPEN 1
#endif
#endif

#if defined(HPX_MSVC) || defined(HPX_MINGW)
#include <hpx/plugin/detail/dll_windows.hpp>
#elif defined(HPX_HAS_DLOPEN)
#include <hpx/plugin/detail/dll_dlopen.hpp>
#else
#error "Hpx.Plugin: your platform is not supported by this library."
#endif
