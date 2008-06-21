//  Copyright (c) 2007-2008 Anshul Tandon
//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/include/applier.hpp>
#include <hpx/components/continuation_impl.hpp>
#include <hpx/lcos/simple_future.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace applier
{
    // 
    lcos::simple_future<naming::id_type> 
    applier::create_async(
        naming::id_type const& targetgid, components::component_type type,
        std::size_t count)
    {
        // Create a simple_future, execute the required action and wait 
        // for the result to be returned to the future.
        lcos::simple_future<naming::id_type> lco;

        // The simple_future instance is associated with the following 
        // apply action by sending it along as its continuation
        apply<components::server::factory::create_action>(
            new components::continuation(lco.get_gid(*this)), targetgid, 
            type, count);

        // we simply return the initialized simple_future, the caller needs
        // to call get_result() on the return value to obtain the result
        return lco;
    }

    // 
    naming::id_type 
    applier::create(threadmanager::px_thread_self& self,
        naming::id_type const& targetgid, components::component_type type,
        std::size_t count)
    {
        return create_async(targetgid, type, count).get_result(self);
    }

///////////////////////////////////////////////////////////////////////////////
}}

