//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_FUNCTIONAL_COMPONENT_NOV_05_2008_0357PM)
#define HPX_COMPONENTS_AMR_FUNCTIONAL_COMPONENT_NOV_05_2008_0357PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/components/amr/stubs/functional_component.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr 
{
    ///////////////////////////////////////////////////////////////////////////
    class functional_component
      : public client_base<functional_component, amr::stubs::functional_component>
    {
    private:
        typedef 
            client_base<functional_component, amr::stubs::functional_component>
        base_type;

    public:
        functional_component(applier::applier& app, naming::id_type gid,
                bool freeonexit = false)
          : base_type(app, gid, freeonexit)
        {}


        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        // The eval and is_last_timestep functions have to be overloaded by any
        // functional component derived from this class
        lcos::future_value<bool> eval_async(naming::id_type const& result, 
            std::vector<naming::id_type> const& gids)
        {
            return this->base_type::eval_async(this->gid_, result, gids);
        }

        bool eval(threads::thread_self& self, naming::id_type const& result, 
            std::vector<naming::id_type> const& gids)
        {
            return this->base_type::eval(self, this->gid_, result, gids);
        }

        ///////////////////////////////////////////////////////////////////////
        lcos::future_value<naming::id_type> alloc_data_async(int item = -1)
        {
            return this->base_type::alloc_data_async(this->gid_, item);
        }

        naming::id_type alloc_data(threads::thread_self& self, int item = -1)
        {
            return this->base_type::alloc_data(self, this->gid_, item);
        }

        ///////////////////////////////////////////////////////////////////////
        void free_data(naming::id_type const& val)
        {
            this->base_type::free_data(this->gid_, val);
        }
    };

}}}

#endif
