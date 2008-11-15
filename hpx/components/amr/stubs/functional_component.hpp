//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_STUBS_FUNCTIONAL_COMPONENT_NOV_05_2008_0338PM)
#define HPX_COMPONENTS_AMR_STUBS_FUNCTIONAL_COMPONENT_NOV_05_2008_0338PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/components/amr/server/functional_component.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace stubs 
{
    ///////////////////////////////////////////////////////////////////////////
    class functional_component
      : public components::stubs::stub_base<amr::server::functional_component>
    {
    private:
        typedef 
            components::stubs::stub_base<amr::server::functional_component> 
        base_type;

    public:
        functional_component(applier::applier& appl)
          : base_type(appl)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        // The eval and is_last_timestep functions have to be overloaded by any
        // functional component derived from this class
        static lcos::future_value<bool> eval_async(
            applier::applier& appl, naming::id_type const& gid, 
            naming::id_type const& result, 
            std::vector<naming::id_type> const& gids)
        {
            // Create an eager_future, execute the required action,
            // we simply return the initialized future_value, the caller needs
            // to call get() on the return value to obtain the result
            typedef amr::server::functional_component::eval_action action_type;
            return lcos::eager_future<action_type>(appl, gid, result, gids);
        }

        static bool eval(threads::thread_self& self, 
            applier::applier& appl, naming::id_type const& gid, 
            naming::id_type const& result, std::vector<naming::id_type> const& gids)
        {
            // The following get yields control while the action above 
            // is executed and the result is returned to the eager_future
            return eval_async(appl, gid, result, gids).get(self);
        }

        lcos::future_value<bool> eval_async(naming::id_type const& gid, 
            naming::id_type const& result, 
            std::vector<naming::id_type> const& gids)
        {
            return eval_async(this->appl_, gid, result, gids);
        }

        bool eval(threads::thread_self& self, naming::id_type const& gid, 
            naming::id_type const& result, std::vector<naming::id_type> const& gids)
        {
            return eval(self, this->appl_, gid, result, gids);
        }

        ///////////////////////////////////////////////////////////////////////
        static lcos::future_value<naming::id_type> alloc_data_async(
            applier::applier& appl, naming::id_type const& gid, int item = -1,
            int maxitems = -1)
        {
            // Create an eager_future, execute the required action,
            // we simply return the initialized future_value, the caller needs
            // to call get() on the return value to obtain the result
            typedef amr::server::functional_component::alloc_data_action action_type;
            return lcos::eager_future<action_type>(appl, gid, item, maxitems);
        }

        static naming::id_type alloc_data(threads::thread_self& self, 
            applier::applier& appl, naming::id_type const& gid, int item = -1,
            int maxitems = -1)
        {
            return alloc_data_async(appl, gid, item, maxitems).get(self);
        }

        lcos::future_value<naming::id_type> alloc_data_async(
            naming::id_type const& gid, int item = -1, int maxitems = -1)
        {
            return alloc_data_async(this->appl_, gid, item, maxitems);
        }

        naming::id_type alloc_data(threads::thread_self& self, 
            naming::id_type const& gid, int item = -1, int maxitems = -1)
        {
            return alloc_data(self, this->appl_, gid, item, maxitems);
        }

        ///////////////////////////////////////////////////////////////////////
        static void free_data(applier::applier& appl, 
            naming::id_type const& gid, naming::id_type const& val)
        {
            typedef amr::server::functional_component::free_data_action action_type;
            appl.apply<action_type>(gid, val);
        }

        void free_data(naming::id_type const& gid, naming::id_type const& val)
        {
            free_data(this->appl_, gid, val);
        }

        static void free_data_sync(threads::thread_self& self, 
            applier::applier& appl, naming::id_type const& gid, 
            naming::id_type const& val)
        {
            typedef amr::server::functional_component::free_data_action action_type;
            lcos::eager_future<action_type>(appl, gid, val).get(self);
        }

        void free_data_sync(threads::thread_self& self, 
            naming::id_type const& gid, naming::id_type const& val)
        {
            free_data_sync(self, this->appl_, gid, val);
        }

        ///////////////////////////////////////////////////////////////////////
        static void init(applier::applier& appl, naming::id_type const& gid, 
            std::size_t numsteps, naming::id_type const& val)
        {
            typedef amr::server::functional_component::init_action action_type;
            appl.apply<action_type>(gid, numsteps, val);
        }

        void init(naming::id_type const& gid, std::size_t numsteps, naming::id_type const& val)
        {
            init(this->appl_, gid, numsteps, val);
        }

        static void init_sync(threads::thread_self& self, 
            applier::applier& appl, naming::id_type const& gid, 
            std::size_t numsteps, naming::id_type const& val)
        {
            typedef amr::server::functional_component::init_action action_type;
            lcos::eager_future<action_type>(appl, gid, numsteps, val).get(self);
        }

        void init_sync(threads::thread_self& self, naming::id_type const& gid, 
            std::size_t numsteps, naming::id_type const& val)
        {
            init_sync(self, this->appl_, gid, numsteps, val);
        }
    };

}}}}

#endif
