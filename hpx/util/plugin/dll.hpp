// Copyright (c) 2005-2013 Hartmut Kaiser
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_PLUGIN_DLL_HPP
#define HPX_UTIL_PLUGIN_DLL_HPP

#include <hpx/config.hpp>

#ifndef HPX_MSVC
# ifndef  HPX_HAS_DLOPEN
#  define HPX_HAS_DLOPEN 1
# endif
#endif

#if defined(HPX_MSVC) || defined(HPX_WINDOWS)
#include <hpx/util/plugin/detail/dll_windows.hpp>
#elif defined(HPX_HAS_DLOPEN)
#include <hpx/util/plugin/detail/dll_dlopen.hpp>
#else
#error "Hpx.Plugin: your platform is not supported by this library."
#endif

#endif /*HPX_UTIL_PLUGIN_DLL_HPP*/
