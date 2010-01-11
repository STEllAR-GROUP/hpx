//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_MESH_TAPERED_AUTONOMOUS_JAN_11_2011_1143AM)
#define HPX_COMPONENTS_AMR_MESH_TAPERED_AUTONOMOUS_JAN_11_2011_1143AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include "stubs/tapered_autonomous.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr 
{
    ///////////////////////////////////////////////////////////////////////////
    class tapered_autonomous
      : public client_base<tapered_autonomous, amr::stubs::tapered_autonomous>
    {
    private:
        typedef 
            client_base<tapered_autonomous, amr::stubs::tapered_autonomous>
        base_type;

    public:
        tapered_autonomous()
          : base_type(naming::invalid_id, false)
        {}
        tapered_autonomous(naming::id_type gid, bool freeonexit = false)
          : base_type(gid, freeonexit)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        // The eval and is_last_timestep functions have to be overloaded by any
        // functional component derived from this class
        lcos::future_value<std::vector<naming::id_type> > 
        init_execute_async(components::component_type function_type, 
            components::component_type logging_type, std::size_t level, double x, Parameter const& par)
        {
            return this->base_type::init_execute_async(this->gid_, function_type,
                logging_type,level, x, par);
        }

        std::vector<naming::id_type> 
        init_execute(components::component_type function_type, 
            components::component_type logging_type, std::size_t level, double x, Parameter const& par)
        {
            return this->base_type::init_execute(this->gid_, function_type,
                logging_type,level,x,par);
        }

        // The eval and is_last_timestep functions have to be overloaded by any
        // functional component derived from this class
        lcos::future_value<std::vector<naming::id_type> > 
        execute_async(std::vector<double> const& initial_data,
            components::component_type function_type, 
            components::component_type logging_type, Parameter const& par)
        {
            return this->base_type::execute_async(this->gid_, initial_data, 
                function_type, logging_type,par);
        }

        std::vector<naming::id_type> 
        execute(std::vector<double> const& initial_data,
            components::component_type function_type, 
            components::component_type logging_type, Parameter const& par)
        {
            return this->base_type::execute(this->gid_, initial_data, 
                function_type, logging_type,par);
        }
    };

}}}

#endif
