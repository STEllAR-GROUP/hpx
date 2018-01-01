//  Copyright (c) 2018 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPILER_NATIVE_TLS)
#define HPX_COMPILER_NATIVE_TLS

#include <hpx/config/defines.hpp>

#if defined(__has_feature)
#  if __has_feature(cxx_thread_local)
#    define HPX_NATIVE_TLS thread_local
#  endif
#elif defined(HPX_HAVE_CXX11_THREAD_LOCAL)
#  define HPX_NATIVE_TLS thread_local
#endif

#if !defined(HPX_NATIVE_TLS)
#  if defined(_GLIBCXX_HAVE_TLS)
#    define HPX_NATIVE_TLS __thread
#  elif defined(HPX_WINDOWS)
#    define HPX_NATIVE_TLS __declspec(thread)
#  elif defined(__FreeBSD__) || (defined(__APPLE__) && defined(__MACH__))
#    define HPX_NATIVE_TLS __thread
#  elif defined(__clang__) && defined(HPX_COMPUTE_DEVICE_CODE)
#    define HPX_NATIVE_TLS __thread
#  else
#    error "Native thread local storage is not supported for this platform, please undefine HPX_HAVE_NATIVE_TLS"
#  endif
#endif

#endif
