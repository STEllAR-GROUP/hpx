//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(REFCNT_TEST_COMPONENTS_STUBS_REFCNT_FEB_08_2010_1057AM)
#define REFCNT_TEST_COMPONENTS_STUBS_REFCNT_FEB_08_2010_1057AM

#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/lcos/eager_future.hpp>

#include "../server/refcnt.hpp"

namespace hpx { namespace components { namespace refcnt_test { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    struct refcnt : components::stubs::stub_base<server::refcnt>
    {
        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component
        static void test(naming::id_type gid) 
        {
            applier::apply<server::refcnt::test_action>(gid);
        }
    };
}}}}

#endif


