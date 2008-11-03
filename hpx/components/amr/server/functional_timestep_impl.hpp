//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_FUNCTIONAL_TIMESTEP__IMPL_OCT_20_2008_1002AM)
#define HPX_COMPONENTS_AMR_FUNCTIONAL_TIMESTEP__IMPL_OCT_20_2008_1002AM

#include <hpx/components/amr/server/functional_timestep.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace server 
{
    ///////////////////////////////////////////////////////////////////////////
    inline functional_timestep::functional_timestep(applier::applier& appl)
      : simple_component_base<functional_timestep>(appl)
    {
    }

    ///////////////////////////////////////////////////////////////////////////
    inline threads::thread_state 
    functional_timestep::initialize(threads::thread_self&, applier::applier&)
    {
        return threads::terminated;
    }

    ///////////////////////////////////////////////////////////////////////////
    inline threads::thread_state 
    functional_timestep::execute(threads::thread_self&, applier::applier&)
    {
        return threads::terminated;
    }

}}}}

#endif

