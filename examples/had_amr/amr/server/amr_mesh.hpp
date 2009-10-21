//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_SERVER_MESH_FEB_16_2009_0834AM)
#define HPX_COMPONENTS_AMR_SERVER_MESH_FEB_16_2009_0834AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/components/distributing_factory/distributing_factory.hpp>

#include "../../amr_client.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace server 
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT amr_mesh
      : public simple_component_base<amr_mesh>
    {
    private:
        typedef simple_component_base<amr_mesh> base_type;

    public:
        amr_mesh();

        // components must contain a typedef for wrapping_type defining the
        // component type used to encapsulate instances of this component
        typedef amr::server::amr_mesh wrapping_type;

        ///////////////////////////////////////////////////////////////////////
        // parcel action code: the action to be performed on the destination 
        // object (the accumulator)
        enum actions
        {
            amr_mesh_init_execute = 0,
            amr_mesh_execute = 1
        };

        /// This is the main entry point of this component. 
        std::vector<naming::id_type> init_execute(
            components::component_type function_type, std::size_t numvalues, 
            std::size_t numsteps,
            components::component_type logging_type,Parameter const& par);

        std::vector<naming::id_type> execute(
            std::vector<naming::id_type> const& initialdata,
            components::component_type function_type, std::size_t numvalues, 
            std::size_t numsteps,
            components::component_type logging_type,Parameter const& par);

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::result_action5<
            amr_mesh, std::vector<naming::id_type>, amr_mesh_init_execute, 
            components::component_type, std::size_t, std::size_t,
            components::component_type,Parameter const&, &amr_mesh::init_execute
        > init_execute_action;

        typedef hpx::actions::result_action6<
            amr_mesh, std::vector<naming::id_type>, amr_mesh_execute, 
            std::vector<naming::id_type> const&,
            components::component_type, std::size_t, std::size_t,
            components::component_type, Parameter const&, &amr_mesh::execute
        > execute_action;

    protected:
        typedef 
            components::distributing_factory::iterator_range_type
        distributed_iterator_range_type;

        static void init(distributed_iterator_range_type const& functions,
            distributed_iterator_range_type const& logging,
            std::size_t numsteps);

        void prepare_initial_data(
            distributed_iterator_range_type const& functions, 
            std::vector<naming::id_type>& initial_data);

        static void init_stencils(
            distributed_iterator_range_type const& stencils,
            distributed_iterator_range_type const& functions, int static_step, int stencilsize,int numvalues,Parameter const& par);

        static void get_output_ports(
            distributed_iterator_range_type const& stencils,
            std::vector<std::vector<naming::id_type> >& outputs);

        static void connect_input_ports(
            components::distributing_factory::result_type const* stencils,
            std::vector<std::vector<std::vector<naming::id_type> > > const& outputs);

        static void execute(distributed_iterator_range_type const& stencils, 
            std::vector<naming::id_type> const& initial_data, 
            std::vector<naming::id_type>& result_data);

        static void start_row(distributed_iterator_range_type const& stencils);

    private:
        std::size_t numvalues_;
    };

}}}}

#endif
