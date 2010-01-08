//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_MESH_LEFT_DEC_4_2009_1120AM)
#define HPX_COMPONENTS_AMR_MESH_LEFT_DEC_4_2009_1120AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include "stubs/amr_mesh_left.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr 
{
    ///////////////////////////////////////////////////////////////////////////
    class amr_mesh_left
      : public client_base<amr_mesh_left, amr::stubs::amr_mesh_left>
    {
    private:
        typedef 
            client_base<amr_mesh_left, amr::stubs::amr_mesh_left>
        base_type;

    public:
        amr_mesh_left()
          : base_type(naming::invalid_id, false)
        {}
        amr_mesh_left(naming::id_type gid, bool freeonexit = false)
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
        execute_async(std::vector<naming::id_type> const& initial_data,
            components::component_type function_type, 
            components::component_type logging_type, Parameter const& par)
        {
            return this->base_type::execute_async(this->gid_, initial_data, 
                function_type, logging_type,par);
        }

        std::vector<naming::id_type> 
        execute(std::vector<naming::id_type> const& initial_data,
            components::component_type function_type, 
            components::component_type logging_type, Parameter const& par)
        {
            return this->base_type::execute(this->gid_, initial_data, 
                function_type, logging_type,par);
        }
    };

}}}

#endif
