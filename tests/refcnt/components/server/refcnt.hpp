//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(REFCNT_TEST_COMPONENTS_SERVER_REFCNT_JAN_25_2010_0955AM)
#define REFCNT_TEST_COMPONENTS_SERVER_REFCNT_JAN_25_2010_0955AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace refcnt_test { namespace server
{
    class refcnt : public simple_component_base<refcnt>
    {
    public:
        // parcel action code: the action to be performed on the destination 
        // object (the accumulator)
        enum actions
        {
            test_refcnt = 0,
        };

        refcnt() {}

        void test() 
        {
        }

        typedef hpx::actions::action0<
            refcnt, test_refcnt, &refcnt::test
        > test_action;
    };
}}}}

#endif
