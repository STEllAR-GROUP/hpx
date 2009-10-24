//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_FUNCTIONAL_COMPONENT_NOV_05_2008_0357PM)
#define HPX_COMPONENTS_AMR_FUNCTIONAL_COMPONENT_NOV_05_2008_0357PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include "stubs/functional_component.hpp"

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
        functional_component(naming::id_type gid, bool freeonexit = false)
          : base_type(gid, freeonexit)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        // The eval and is_last_timestep functions have to be overloaded by any
        // functional component derived from this class
        lcos::future_value<int> eval_async(naming::id_type const& result, 
            std::vector<naming::id_type> const& gids, int row, int column,
            Parameter const& par)
        {
            return this->base_type::eval_async(this->gid_, result, gids, row, column,par);
        }

        int eval(naming::id_type const& result, 
            std::vector<naming::id_type> const& gids, int row, int column,
            Parameter const& par)
        {
            return this->base_type::eval(this->gid_, result, gids, row, column,par);
        }

        ///////////////////////////////////////////////////////////////////////
        lcos::future_value<naming::id_type> alloc_data_async(int item,
            int maxitems, int row, Parameter const& par)
        {
            return this->base_type::alloc_data_async(this->gid_, item, 
                maxitems, row, par);
        }

        naming::id_type alloc_data(int item, int maxitems,
            int row, Parameter const& par)
        {
            return this->base_type::alloc_data(this->gid_, item, maxitems, 
                row, par);
        }

        ///////////////////////////////////////////////////////////////////////
        void init(std::size_t numsteps, naming::id_type const& val)
        {
            this->base_type::init(this->gid_, numsteps, val);
        }

        void init_sync(std::size_t numsteps, naming::id_type const& val)
        {
            this->base_type::init_sync(this->gid_, numsteps, val);
        }
    };

}}}

#endif
