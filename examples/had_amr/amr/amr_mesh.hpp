//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_MESH_FEB_16_2009_0229PM)
#define HPX_COMPONENTS_AMR_MESH_FEB_16_2009_0229PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include "stubs/amr_mesh.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr 
{
    ///////////////////////////////////////////////////////////////////////////
    class amr_mesh
      : public client_base<amr_mesh, amr::stubs::amr_mesh>
    {
    private:
        typedef 
            client_base<amr_mesh, amr::stubs::amr_mesh>
        base_type;

    public:
        amr_mesh()
        {}

        amr_mesh(naming::id_type const& gid)
          : base_type(gid)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        // The eval and is_last_timestep functions have to be overloaded by any
        // functional component derived from this class
        lcos::future_value<std::vector<naming::id_type> > 
        init_execute_async(components::component_type function_type, 
            std::size_t numvalues, std::size_t numsteps,
           // components::component_type logging_type = components::component_invalid,
            components::component_type logging_type, Parameter const& par)
        {
            return this->base_type::init_execute_async(this->gid_, function_type,
                numvalues, numsteps, logging_type,par);
        }

        std::vector<naming::id_type> 
        init_execute(components::component_type function_type, 
            std::size_t numvalues, std::size_t numsteps,
            components::component_type logging_type, Parameter const& par)
        {
            return this->base_type::init_execute(this->gid_, function_type,
                numvalues, numsteps,logging_type,par);
        }

        // The eval and is_last_timestep functions have to be overloaded by any
        // functional component derived from this class
        lcos::future_value<std::vector<naming::id_type> > 
        execute_async(std::vector<naming::id_type> const& initial_data,
            components::component_type function_type, 
            std::size_t numvalues, std::size_t numsteps,
            components::component_type logging_type, Parameter const& par)
        {
            return this->base_type::execute_async(this->gid_, initial_data, 
                function_type, numvalues, numsteps, logging_type,par);
        }

        std::vector<naming::id_type> 
        execute(std::vector<naming::id_type> const& initial_data,
            components::component_type function_type, 
            std::size_t numvalues, std::size_t numsteps,
            components::component_type logging_type, Parameter const& par)
        {
            return this->base_type::execute(this->gid_, initial_data, 
                function_type, numvalues, numsteps, logging_type,par);
        }
    };

}}}

#endif
