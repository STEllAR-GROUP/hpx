////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_602CD6B7_9C5C_4FB7_B558_456B04AC6948)
#define HPX_602CD6B7_9C5C_4FB7_B558_456B04AC6948

#include <boost/noncopyable.hpp>

#include <hpx/runtime/agas/client/legacy/user_base.hpp>
#include <hpx/runtime/agas/client/legacy/agent_base.hpp>

namespace hpx { namespace agas { namespace legacy
{

template <typename Database>
struct user_agent : agent_base<user_base<Database> >, boost::noncopyable
{
    typedef agent_base<user_base<Database> > base_type;

    user_agent(util::runtime_configuration const& ini_
                  = util::runtime_configuration(), 
               runtime_mode mode = runtime_mode_worker,
               agent_state as_state = agent_state_active)
        : base_type(ini_, mode, as_state)
    {
        BOOST_ASSERT(as_state != agent_state_bootstrapping);
        BOOST_ASSERT(as_state != agent_state_invalid);
    } 
};

}}}

#endif // HPX_602CD6B7_9C5C_4FB7_B558_456B04AC6948

