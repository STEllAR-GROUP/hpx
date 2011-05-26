////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_E81B45FB_3456_4454_98E7_95B789D6ED86)
#define HPX_E81B45FB_3456_4454_98E7_95B789D6ED86

#include <boost/assert.hpp>
#include <boost/noncopyable.hpp>

#include <hpx/runtime/agas/client/legacy/bootstrap_base.hpp>
#include <hpx/runtime/agas/client/legacy/agent_base.hpp>

namespace hpx { namespace agas { namespace legacy
{

template <typename Database>
struct bootstrap_agent : agent_base<bootstrap_base<Database> >
                       , boost::noncopyable
{
    typedef agent_base<bootstrap_base<Database> > base_type;

    bootstrap_agent(util::runtime_configuration const& ini_
                      = util::runtime_configuration(), 
                    runtime_mode mode = runtime_mode_worker,
                    agent_state as_state = agent_state_bootstrapping)
        : base_type(ini_, mode, as_state)
    { BOOST_ASSERT(as_state != agent_state_invalid); }
};

}}}

#endif // HPX_E81B45FB_3456_4454_98E7_95B789D6ED86

