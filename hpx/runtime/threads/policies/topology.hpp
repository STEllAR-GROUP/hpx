//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADS_POLICIES_TOPOLGY_NOV_25_2012_1036AM)
#define HPX_THREADS_POLICIES_TOPOLGY_NOV_25_2012_1036AM

#include <hpx/hpx_fwd.hpp>

#if defined(HPX_HAVE_HWLOC)
#  include <hpx/runtime/threads/policies/hwloc_topology.hpp>
#elif defined(BOOST_WINDOWS)
#  include <hpx/runtime/threads/policies/windows_topology.hpp>
#elif defined(__APPLE__)
#  include <hpx/runtime/threads/policies/macosx_topology.hpp>
#elif defined(__linux__) && !defined(__ANDROID__) && !defined(ANDROID)
#  include <hpx/runtime/threads/policies/linux_topology.hpp>
#else
#  include <hpx/runtime/threads/policies/noop_topology.hpp>
#endif

#endif
