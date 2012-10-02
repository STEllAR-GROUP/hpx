//  Copyright (c) 2007-2010 Hartmut Kaiser
//  Copyright (c) 2009-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_STUBS_RK_MESH_FEB_16_2010_0313PM)
#define HPX_COMPONENTS_AMR_STUBS_RK_MESH_FEB_25_2010_0313PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>

#include "../server/unigrid_mesh.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    struct unigrid_mesh
      : components::stub_base<amr::server::unigrid_mesh>
    {
        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        ///////////////////////////////////////////////////////////////////////
        static lcos::future<boost::shared_ptr<std::vector<naming::id_type> > >
        init_execute_async(naming::id_type const& gid,
            std::vector<naming::id_type> const& interp_src_data,
            double time,
            components::component_type function_type, std::size_t numvalues,
            std::size_t numsteps, components::component_type logging_type,
            parameter const& par)
        {
            // Create a future, execute the required action,
            // we simply return the initialized future, the caller needs
            // to call get() on the return value to obtain the result
            typedef amr::server::unigrid_mesh::init_execute_action action_type;
            return hpx::async<action_type>(gid, interp_src_data,time,function_type,
                numvalues, numsteps, logging_type,par);
        }

        static boost::shared_ptr<std::vector<naming::id_type> >
        init_execute(naming::id_type const& gid,
            std::vector<naming::id_type> const& interp_src_data,
            double time,
            components::component_type function_type, std::size_t numvalues,
            std::size_t numsteps, components::component_type logging_type,
            parameter const& par)
        {
            return init_execute_async(gid, interp_src_data,time,
                function_type, numvalues, numsteps,
                logging_type,par).get();
        }

        ///////////////////////////////////////////////////////////////////////
        static lcos::future<std::vector<naming::id_type> >
        execute_async(naming::id_type const& gid,
            std::vector<naming::id_type> const& initial_data,
            components::component_type function_type, std::size_t numvalues,
            std::size_t numsteps, components::component_type logging_type,
            parameter const& par)
        {
            // Create a future, execute the required action,
            // we simply return the initialized future, the caller needs
            // to call get() on the return value to obtain the result
            typedef amr::server::unigrid_mesh::execute_action action_type;
            return hpx::async<action_type>(gid, initial_data,
                function_type, numvalues, numsteps, logging_type, par);
        }

        static std::vector<naming::id_type> execute(naming::id_type const& gid,
            std::vector<naming::id_type> const& initial_data,
            components::component_type function_type, std::size_t numvalues,
            std::size_t numsteps, components::component_type logging_type,
            parameter const& par)
        {
            return execute_async(gid, initial_data, function_type, numvalues,
                numsteps, logging_type,par).get();
        }
    };

}}}}

#endif
