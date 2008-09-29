// Copyright Hartmut Kaiser 2005.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_DLL_HPP_HK_2005_11_06
#define BOOST_DLL_HPP_HK_2005_11_06

#include <boost/config.hpp>

#ifndef BOOST_WINDOWS
# ifndef  BOOST_HAS_DLOPEN
#  define BOOST_HAS_DLOPEN 1
# endif
#endif

#if defined(BOOST_WINDOWS)
#include <boost/plugin/detail/dll_windows.hpp>
#elif defined(BOOST_HAS_DLOPEN)
#include <boost/plugin/detail/dll_dlopen.hpp>
#else
#error "Boost.Plugin: your platform is not supported by this library."
#endif

#endif
