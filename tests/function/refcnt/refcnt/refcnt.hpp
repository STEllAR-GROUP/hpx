//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(REFCNT_TEST_COMPONENTS_REFCNT_JAN_25_2010_0955AM)
#define REFCNT_TEST_COMPONENTS_REFCNT_JAN_25_2010_0955AM

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include "stubs/refcnt.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace refcnt_test
{
    ///////////////////////////////////////////////////////////////////////////
    class refcnt : public client_base<refcnt, stubs::refcnt>
    {
        typedef client_base<refcnt, stubs::refcnt> base_type;

    public:
        refcnt() 
          : base_type(naming::invalid_id)
        {}

        /// Create a client side representation for the existing
        /// \a server#simple_accumulator instance with the given global id \a gid.
        refcnt(naming::id_type gid) 
          : base_type(gid)
        {}

        ~refcnt() 
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component
        void test() 
        {
            BOOST_ASSERT(gid_);
            this->base_type::test(gid_);
        }
    };
}}}

#endif
