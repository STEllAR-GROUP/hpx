//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2009-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_RK_MESH_FEB_25_2011_0312PM)
#define HPX_COMPONENTS_RK_MESH_FEB_25_2011_0312PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include "stubs/dataflow_stencil.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace adaptive1d
{
    ///////////////////////////////////////////////////////////////////////////
    class dataflow_stencil
      : public client_base<dataflow_stencil, adaptive1d::stubs::dataflow_stencil>
    {
    private:
        typedef
            client_base<dataflow_stencil, adaptive1d::stubs::dataflow_stencil>
        base_type;

    public:
        dataflow_stencil()
        {}

        dataflow_stencil(naming::id_type gid)
          : base_type(gid)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        // The eval and is_last_timestep functions have to be overloaded by any
        // functional component derived from this class
        lcos::promise<boost::shared_ptr<std::vector<naming::id_type> > >
        init_execute_async(std::vector<naming::id_type> const& interp_src_data,
            double time,
            components::component_type function_type,
            std::size_t numvalues, std::size_t numsteps,
           // components::component_type logging_type = components::component_invalid,
            components::component_type logging_type,
            parameter const& par)
        {
            return this->base_type::init_execute_async(this->gid_,
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
            return this->base_type::init_execute(this->gid_,
                interp_src_data,time, function_type,
                numvalues, numsteps,logging_type,par);
        }

        // The eval and is_last_timestep functions have to be overloaded by any
        // functional component derived from this class
        lcos::promise<std::vector<naming::id_type> >
        execute_async(std::vector<naming::id_type> const& initial_data,
            components::component_type function_type,
            std::size_t numvalues, std::size_t numsteps,
            components::component_type logging_type, parameter const& par)
        {
            return this->base_type::execute_async(this->gid_, initial_data,
                function_type, numvalues, numsteps, logging_type,par);
        }

        std::vector<naming::id_type>
        execute(std::vector<naming::id_type> const& initial_data,
            components::component_type function_type,
            std::size_t numvalues, std::size_t numsteps,
            components::component_type logging_type, parameter const& par)
        {
            return this->base_type::execute(this->gid_, initial_data,
                function_type, numvalues, numsteps, logging_type,par);
        }
    };

}}}

#endif
