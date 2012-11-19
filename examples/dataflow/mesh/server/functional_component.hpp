//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2009-2011 Matt Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_SERVER_FUNCTIONAL_COMPONENT_OCT_19_2008_1234PM)
#define HPX_COMPONENTS_AMR_SERVER_FUNCTIONAL_COMPONENT_OCT_19_2008_1234PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>

#include "../../parameter.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT functional_component
    {
    public:
        functional_component()
        {
            //if (component_invalid == base_type::get_component_type()) {
            //    // first call to get_component_type, ask AGAS for a unique id
            //    base_type::set_component_type(applier::get_applier().
            //        get_agas_client().get_component_id("dataflow_functional_component_type"));
            //}
        }

        // components must contain a typedef for wrapping_type defining the
        // managed_component type used to encapsulate instances of this
        // component
        typedef managed_component<functional_component> wrapping_type;
        typedef functional_component base_type_holder;

        static components::component_type get_component_type()
        {
            return components::get_component_type<wrapping_type>();
        }
        static void set_component_type(components::component_type t)
        {
            components::set_component_type<wrapping_type>(t);
        }

        // The eval and is_last_timestep functions have to be overloaded by any
        // functional component derived from this class
        virtual int eval(naming::id_type const&,
            std::vector<naming::id_type> const&, std::size_t, std::size_t,
            double,parameter const&)
        {
            // This shouldn't ever be called. If you're seeing this assertion
            // you probably forgot to overload this function in your stencil
            // class.
            BOOST_ASSERT(false);
            return true;
        }

        virtual naming::id_type alloc_data(int item, int maxitems, int row,
            std::vector<naming::id_type> const& interp_src_data,
            double time,
            parameter const&)
        {
            // This shouldn't ever be called. If you're seeing this assertion
            // you probably forgot to overload this function in your stencil
            // class.
            BOOST_ASSERT(false);
            return naming::invalid_id;
        }

        virtual void init(std::size_t, naming::id_type const&)
        {
            // This shouldn't ever be called. If you're seeing this assertion
            // you probably forgot to overload this function in your stencil
            // class.
            BOOST_ASSERT(false);
        }

        /// This is the main entry point of this component. Calling this
        /// function (by applying the eval_action) will compute the next
        /// time step value based on the result values of the previous time
        /// steps.
        int eval_nonvirt(naming::id_type const& result,
            std::vector<naming::id_type> const& gids, std::size_t row,
            std::size_t column,double cycle_time, parameter const& par)
        {
            return eval(result, gids, row, column,cycle_time,par);
        }

        naming::id_type alloc_data_nonvirt(int item, int maxitems, int row,
            std::vector<naming::id_type> const& interp_src_data,
            double time,
            parameter const& par)
        {
            return alloc_data(item, maxitems, row, interp_src_data,time,par);
        }

        util::unused_type
        init_nonvirt(std::size_t numsteps, naming::id_type const& gid)
        {
            init(numsteps, gid);
            return util::unused;
        }

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        HPX_DEFINE_COMPONENT_ACTION(functional_component, alloc_data_nonvirt, alloc_data_action);
        HPX_DEFINE_COMPONENT_ACTION(functional_component, eval_nonvirt, eval_action);
        HPX_DEFINE_COMPONENT_ACTION(functional_component, init_nonvirt, init_action);

        /// This is the default hook implementation for decorate_action which 
        /// does no hooking at all.
        static HPX_STD_FUNCTION<threads::thread_function_type> 
        wrap_action(HPX_STD_FUNCTION<threads::thread_function_type> f,
            naming::address::address_type)
        {
            return boost::move(f);
        }
    };
}}}}

HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::amr::server::functional_component::alloc_data_action,
    dataflow_functional_component_alloc_data_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::amr::server::functional_component::eval_action,
    dataflow_functional_component_eval_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::amr::server::functional_component::init_action,
    dataflow_functional_component_init_action);

#endif
