//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2009-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_RK_MESH_FEB_25_2010_0312PM)
#define HPX_COMPONENTS_RK_MESH_FEB_25_2010_0312PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/include/client.hpp>

#include "stubs/unigrid_mesh.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr
{
    ///////////////////////////////////////////////////////////////////////////
    class unigrid_mesh
      : public client_base<unigrid_mesh, amr::stubs::unigrid_mesh>
    {
    private:
        typedef
            client_base<unigrid_mesh, amr::stubs::unigrid_mesh>
        base_type;

    public:
        unigrid_mesh()
        {}

        unigrid_mesh(naming::id_type gid)
          : base_type(gid)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        // The eval and is_last_timestep functions have to be overloaded by any
        // functional component derived from this class
        lcos::future<boost::shared_ptr<std::vector<naming::id_type> > >
        init_execute_async(std::vector<naming::id_type> const& interp_src_data,
            double time,
            components::component_type function_type,
            std::size_t numvalues, std::size_t numsteps,
           // components::component_type logging_type = components::component_invalid,
            components::component_type logging_type,
            parameter const& par)
        {
            return this->base_type::init_execute_async(this->get_gid(),
                interp_src_data,time,function_type,
                numvalues, numsteps, logging_type,par);
        }

        boost::shared_ptr<std::vector<naming::id_type> >
        init_execute(std::vector<naming::id_type> const& interp_src_data,
            double time,
            components::component_type function_type,
            std::size_t numvalues, std::size_t numsteps,
            components::component_type logging_type,
            parameter const& par)
        {
            return this->base_type::init_execute(this->get_gid(),
                interp_src_data,time, function_type,
                numvalues, numsteps,logging_type,par);
        }

        // The eval and is_last_timestep functions have to be overloaded by any
        // functional component derived from this class
        lcos::future<std::vector<naming::id_type> >
        execute_async(std::vector<naming::id_type> const& initial_data,
            components::component_type function_type,
            std::size_t numvalues, std::size_t numsteps,
            components::component_type logging_type, parameter const& par)
        {
            return this->base_type::execute_async(this->get_gid(), initial_data,
                function_type, numvalues, numsteps, logging_type,par);
        }

        std::vector<naming::id_type>
        execute(std::vector<naming::id_type> const& initial_data,
            components::component_type function_type,
            std::size_t numvalues, std::size_t numsteps,
            components::component_type logging_type, parameter const& par)
        {
            return this->base_type::execute(this->get_gid(), initial_data,
                function_type, numvalues, numsteps, logging_type,par);
        }
    };

}}}

#endif
