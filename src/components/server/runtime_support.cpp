//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/components/server/runtime_support.hpp>
#include <hpx/components/stubs/runtime_support.hpp>
#include <hpx/components/server/accumulator.hpp>
#include <hpx/components/server/distributing_factory.hpp>
#include <hpx/components/server/manage_component.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>

namespace hpx { namespace components { namespace server
{
    // create a new instance of a component
    threads::thread_state runtime_support::create_component(
        threads::thread_self& self, applier::applier& appl,
        naming::id_type* gid, components::component_type type,
        std::size_t count)
    {
    // create new component instance
        naming::id_type id = naming::invalid_id;
        switch (type) {
        case accumulator::value:
            id = server::create<server::accumulator>(appl, count);
            break;

        case distributing_factory::value:
            id = server::create<server::distributing_factory>(appl, count);
            break;

        default:
            boost::throw_exception(hpx::exception(hpx::bad_component_type,
                std::string("attempt to create component instance of invalid type: ") + 
                    get_component_type_name(type)));
            break;
        }

    // set result if requested
        if (0 != gid)
            *gid = id;
        return threads::terminated;
    }

    // delete an existing instance of a component
    threads::thread_state runtime_support::free_component(
        threads::thread_self& self, applier::applier& appl,
        components::component_type type, naming::id_type const& gid,
        std::size_t count)
    {
        switch (type) {
        case accumulator::value:
            server::destroy<server::accumulator>(appl, gid, count);
            break;

        case distributing_factory::value:
            server::destroy<server::distributing_factory>(appl, gid, count);
            break;

        default:
            boost::throw_exception(hpx::exception(hpx::bad_component_type,
                std::string("attempt to create component instance of invalid type: ") + 
                    get_component_type_name(type)));
            break;
        }
        return threads::terminated;
    }

    /// \brief Action shut down this runtime system instance
    threads::thread_state runtime_support::shutdown(
        threads::thread_self& self, applier::applier& app)
    {
        // initiate system shutdown
        stop();
        return threads::terminated;
    }

    // initiate system shutdown for all localities
    threads::thread_state runtime_support::shutdown_all(
        threads::thread_self& self, applier::applier& app)
    {
        std::vector<naming::id_type> prefixes;
        app.get_dgas_client().get_prefixes(prefixes);

        // shut down all localities except the the local one
        components::stubs::runtime_support rts(app);
        std::vector<naming::id_type>::iterator end = prefixes.end();
        for (std::vector<naming::id_type>::iterator it = prefixes.begin(); 
             it != end; ++it)
        {
            if (naming::get_prefix_from_id(app.get_prefix()) !=
                naming::get_prefix_from_id(*it))
            {
                rts.shutdown(*it);
            }
        }

        // now make sure the local locality gets shut down as well.
        stop();
        return threads::terminated;
    }

    ///////////////////////////////////////////////////////////////////////////
    void runtime_support::wait()
    {
        mutex_type::scoped_lock l(mtx_);
        stopped_ = false;
        condition_.wait(l);
    }

    void runtime_support::stop()
    {
        mutex_type::scoped_lock l(mtx_);
        if (!stopped_) {
            condition_.notify_all();
            stopped_ = true;
        }
    }

}}}

