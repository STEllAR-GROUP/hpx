////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <tests/correctness/agas/components/server/managed_refcnt_checker.hpp>
#include <hpx/runtime/applier/trigger.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>

namespace hpx { namespace test { namespace server
{

managed_refcnt_checker::~managed_refcnt_checker()
{
    if (naming::invalid_id != target_)
    {
        applier::trigger(target_);
    }
}

}}}

