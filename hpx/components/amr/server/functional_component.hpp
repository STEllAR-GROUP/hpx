//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_SERVER_FUNCTIONAL_COMPONENT_OCT_19_2008_1234PM)
#define HPX_COMPONENTS_AMR_SERVER_FUNCTIONAL_COMPONENT_OCT_19_2008_1234PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace server 
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT functional_component
      : public simple_component_base<functional_component>
    {
    private:
        typedef simple_component_base<functional_component> base_type;

    public:
        functional_component()
        {
            if (component_invalid == base_type::get_component_type()) {
                // first call to get_component_type, ask AGAS for a unique id
                base_type::set_component_type(applier::get_applier().
                    get_agas_client().get_component_id("functional_component_type"));
            }
        }

        // components must contain a typedef for wrapping_type defining the
        // managed_component type used to encapsulate instances of this 
        // component
        typedef amr::server::functional_component wrapping_type;

        // The eval and is_last_timestep functions have to be overloaded by any
        // functional component derived from this class
        virtual threads::thread_state eval(bool* islast, naming::id_type const&, 
            std::vector<naming::id_type> const&)
        {
            // This shouldn't ever be called. If you're seeing this assertion 
            // you probably forgot to overload this function in your stencil 
            // class.
            BOOST_ASSERT(false);
            return threads::terminated;
        }

        virtual threads::thread_state alloc_data(naming::id_type*, int item,
            int maxitems)
        {
            // This shouldn't ever be called. If you're seeing this assertion 
            // you probably forgot to overload this function in your stencil 
            // class.
            BOOST_ASSERT(false);
            return threads::terminated;
        }

        virtual threads::thread_state free_data(naming::id_type const&)
        {
            // This shouldn't ever be called. If you're seeing this assertion 
            // you probably forgot to overload this function in your stencil 
            // class.
            BOOST_ASSERT(false);
            return threads::terminated;
        }

        virtual threads::thread_state init(std::size_t, naming::id_type const&)
        {
            // This shouldn't ever be called. If you're seeing this assertion 
            // you probably forgot to overload this function in your stencil 
            // class.
            BOOST_ASSERT(false);
            return threads::terminated;
        }

        ///////////////////////////////////////////////////////////////////////
        // parcel action code: the action to be performed on the destination 
        // object (the accumulator)
        enum actions
        {
            functional_component_alloc_data = 0,
            functional_component_eval = 1,
            functional_component_free_data = 2,
            functional_component_init = 3
        };

        /// This is the main entry point of this component. Calling this 
        /// function (by applying the eval_action) will compute the next 
        /// time step value based on the result values of the previous time 
        /// steps.
        threads::thread_state eval_nonvirt(
            bool* retval, naming::id_type const& result, 
            std::vector<naming::id_type> const& gids)
        {
            return eval(retval, result, gids);
        }

        threads::thread_state alloc_data_nonvirt(
            naming::id_type* result, int item, int maxitems)
        {
            return alloc_data(result, item, maxitems);
        }

        threads::thread_state free_data_nonvirt(naming::id_type const& gid)
        {
            return free_data(gid);
        }

        threads::thread_state init_nonvirt(
            std::size_t numsteps, naming::id_type const& gid)
        {
            return init(numsteps, gid);
        }

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::result_action2<
            functional_component, naming::id_type, functional_component_alloc_data, 
            int, int, &functional_component::alloc_data_nonvirt
        > alloc_data_action;

        typedef hpx::actions::result_action2<
            functional_component, bool, functional_component_eval, 
            naming::id_type const&, std::vector<naming::id_type> const&, 
            &functional_component::eval_nonvirt
        > eval_action;

        typedef hpx::actions::action1<
            functional_component, functional_component_free_data, 
            naming::id_type const&, &functional_component::free_data_nonvirt
        > free_data_action;

        typedef hpx::actions::action2<
            functional_component, functional_component_init, 
            std::size_t, naming::id_type const&, &functional_component::init_nonvirt
        > init_action;
    };

}}}}

#endif
