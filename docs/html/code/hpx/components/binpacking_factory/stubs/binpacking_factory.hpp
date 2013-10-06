//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUBS_BINPACKING_FACTORY_MAY_23_2012_1131AM)
#define HPX_COMPONENTS_STUBS_BINPACKING_FACTORY_MAY_23_2012_1131AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/components/binpacking_factory/server/binpacking_factory.hpp>

namespace hpx { namespace components { namespace stubs
{
    ///////////////////////////////////////////////////////////////////
    // The \a stubs#binpacking_factoryclass is the client side
    // representation of a \a server#binpacking_factorycomponent
    struct binpacking_factory : stub_base<server::binpacking_factory>
    {
        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component
        typedef server::binpacking_factory::result_type result_type;
        typedef server::binpacking_factory::remote_result_type remote_result_type;
        typedef server::binpacking_factory::iterator_type iterator_type;
        typedef server::binpacking_factory::iterator_range_type iterator_range_type;

        ///////////////////////////////////////////////////////////////////////
        /// \brief Create new components of the given type while
        ///        taking into account the current number of instances existing
        ///        on the target localities.
        static lcos::future<result_type>
        create_components_async(
            naming::id_type const& targetgid, components::component_type type,
            std::size_t count)
        {
            // Create a future, execute the required action,
            // we simply return the initialized future, the caller needs
            // to call get() on the return value to obtain the result
            typedef server::binpacking_factory::create_components_action
                action_type;
            return hpx::async<action_type>(targetgid, type, count);
        }

        static result_type create_components(naming::id_type const& targetgid,
            components::component_type type, std::size_t count = 1)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the future
            return create_components_async(targetgid, type, count).get();
        }

        ///////////////////////////////////////////////////////////////////////
        /// \brief Create new components of the given type while
        ///        taking into account the current number of instances existing
        ///        on the target localities.
        static lcos::future<result_type>
        create_components_counterbased_async(
            naming::id_type const& targetgid, components::component_type type,
            std::size_t count, std::string const& countername)
        {
            // Create a future, execute the required action,
            // we simply return the initialized future, the caller needs
            // to call get() on the return value to obtain the result
            typedef server::binpacking_factory::create_components_counterbased_action
                action_type;
            return hpx::async<action_type>(targetgid, type, count, countername);
        }

        static result_type create_components_counterbased(
            naming::id_type const& targetgid, components::component_type type,
            std::size_t count, std::string const& countername)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the future
            return create_components_counterbased_async(
                targetgid, type, count, countername).get();
        }
    };
}}}

#endif
