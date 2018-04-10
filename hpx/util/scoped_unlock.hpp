//  Copyright (c) 2007-2008 Chirag Dekate, Hartmut Kaiser
//  Copyright (c) 2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_SCOPED_UNLOCK_HPP
#define HPX_UTIL_SCOPED_UNLOCK_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_SCOPED_UNLOCK_COMPATIBILITY)
#include <hpx/util/unlock_guard.hpp>

namespace hpx { namespace util
{
    template <typename Lock>
    using scoped_unlock = unlock_guard<Lock>;
}}
#else
#  error This header exists for compatibility reasons, use <hpx/util/unlock_guard.hpp> instead.
#endif

#endif /*HPX_UTIL_SCOPED_UNLOCK_HPP*/
