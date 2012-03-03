//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUBS_DISTRIBUTING_FACTORY_OCT_19_2008_0452PM)
#define HPX_COMPONENTS_STUBS_DISTRIBUTING_FACTORY_OCT_19_2008_0452PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/components/distributing_factory/server/distributing_factory.hpp>

namespace hpx { namespace components { namespace stubs
{
    ///////////////////////////////////////////////////////////////////
    // The \a stubs#distributing_factory class is the client side
    // representation of a \a server#distributing_factory component
    struct distributing_factory : stub_base<server::distributing_factory>
    {
        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component
        typedef server::distributing_factory::result_type result_type;
        typedef server::distributing_factory::remote_result_type remote_result_type;
        typedef server::distributing_factory::iterator_type iterator_type;
        typedef server::distributing_factory::iterator_range_type iterator_range_type;

        /// Create a number of new components of the given \a type distributed
        /// evenly over all available localities. This is a non-blocking call.
        /// The caller needs to call \a promise#get on the result
        /// of this function to obtain the global ids of the newly created
        /// objects.
        static lcos::promise<result_type, remote_result_type>
        create_components_async(
            naming::id_type const& targetgid, components::component_type type,
            std::size_t count)
        {
            // Create an eager_future, execute the required action,
            // we simply return the initialized promise, the caller needs
            // to call get() on the return value to obtain the result
            typedef server::distributing_factory::create_components_action
                action_type;
            return lcos::eager_future<action_type, result_type>(
                targetgid, type, count);
        }

        /// Create a number of new components of the given \a type distributed
        /// evenly over all available localities. Block for the creation to finish.
        static result_type create_components(naming::id_type const& targetgid,
            components::component_type type, std::size_t count = 1)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the eager_future
            return create_components_async(targetgid, type, count).get();
        }

        /// Free components
        static void free_components(naming::id_type const& factory,
            result_type const& gids)
        {
//             typedef server::distributing_factory::free_components_action
//                 action_type;
//             hpx::applier::apply<action_type>(factory, gids, false);
        }

        static void free_components_sync(naming::id_type const& factory,
            result_type const& gids)
        {
//             typedef server::distributing_factory::free_components_action
//                 action_type;
//             lcos::eager_future<action_type, void>(factory, gids, true).get();
        }
    };

}}}

#endif
