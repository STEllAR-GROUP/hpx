//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2012 Hartmut Kaiser
//  Copyright (c) 2009 Oliver Kowalke
//
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#include <hpx/runtime/coroutine/detail/context_base.hpp>

namespace hpx { namespace coroutines { namespace detail
{
    // initialize static allocation counter
    allocation_counters context_base::m_allocation_counters;
}}}
