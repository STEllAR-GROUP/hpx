//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2009-2011 Matt Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_DATAFLOW_SERVER_RK_MESH_FEB_25_2011_0153AM)
#define HPX_COMPONENTS_DATAFLOW_SERVER_RK_MESH_FEB_25_2011_0153AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/components/distributing_factory/distributing_factory.hpp>

#include "../../parameter.hpp"
#include "../../array3d.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace adaptive1d { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT dataflow_stencil
      : public simple_component_base<dataflow_stencil>
    {
    private:
        typedef simple_component_base<dataflow_stencil> base_type;

    public:
        dataflow_stencil();

        // components must contain a typedef for wrapping_type defining the
        // component type used to encapsulate instances of this component
        typedef adaptive1d::server::dataflow_stencil wrapping_type;

        ///////////////////////////////////////////////////////////////////////
        // parcel action code: the action to be performed on the destination
        // object (the accumulator)
        enum actions
        {
            dataflow_stencil_init_execute = 0,
            dataflow_stencil_execute = 1
        };

        /// This is the main entry point of this component.
        boost::shared_ptr<std::vector<naming::id_type> > init_execute(
            std::vector<naming::id_type> const& interp_src_data,
            double time,
            components::component_type function_type, std::size_t numvalues,
            std::size_t numsteps,
            components::component_type logging_type,
            parameter const& par);

        std::vector<naming::id_type> execute(
            std::vector<naming::id_type> const& initialdata,
            components::component_type function_type, std::size_t numvalues,
            std::size_t numsteps,
            components::component_type logging_type, parameter const& par);

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::result_action7<
            dataflow_stencil, boost::shared_ptr<std::vector<naming::id_type> >,
            dataflow_stencil_init_execute,std::vector<naming::id_type> const&,
            double,
            components::component_type,
            std::size_t, std::size_t, components::component_type,
            parameter const&, &dataflow_stencil::init_execute
        > init_execute_action;

        typedef hpx::actions::result_action6<
            dataflow_stencil, std::vector<naming::id_type>, dataflow_stencil_execute,
            std::vector<naming::id_type> const&,
            components::component_type, std::size_t, std::size_t,
            components::component_type, parameter const&, &dataflow_stencil::execute
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
            std::vector<naming::id_type> const& interp_src_data,
            std::vector<naming::id_type>& initial_data,
            double time,
            std::size_t numvalues,
            parameter const& par);

        static void init_stencils(
            distributed_iterator_range_type const& stencils,
            distributed_iterator_range_type const& functions, int static_step,
            array3d &dst_port,array3d &dst_src,array3d &dst_step,
            array3d &dst_size,array3d &src_size,double cycle_time,parameter const& par);

        static void get_output_ports(
            distributed_iterator_range_type const& stencils,
            std::vector<std::vector<naming::id_type> >& outputs);

        static void connect_input_ports(
            components::distributing_factory::result_type const* stencils,
            std::vector<std::vector<std::vector<naming::id_type> > > const& outputs,
            array3d &dst_size,array3d &dst_step,array3d &dst_src,array3d &dst_port,
            parameter const& par);

        static void execute(distributed_iterator_range_type const& stencils,
            std::vector<naming::id_type> const& initial_data,
            std::vector<naming::id_type>& result_data);

        static void start_row(distributed_iterator_range_type const& stencils);

        static void prep_ports(array3d &dst_port,array3d &dst_src,
                                    array3d &dst_step,array3d &dst_size,
                                    array3d &src_size,std::size_t num_rows,
                                    std::vector<std::size_t> &each_row,
                                    parameter const& par);

        static bool intersection(double_type xmin,double_type xmax,
                                    double_type xmin2,double_type xmax2);


        static bool floatcmp(double_type const& x1, double_type const& x2);
        static bool floatcmp_ge(double_type const& x1, double_type const& x2);
        static bool floatcmp_le(double_type const& x1, double_type const& x2);

    private:
    };

}}}}

HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::adaptive1d::server::dataflow_stencil::init_execute_action,
    adaptive1d_dataflow_dataflow_stencil_init_execute_action);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::adaptive1d::server::dataflow_stencil::execute_action,
    adaptive1d_dataflow_dataflow_stencil_execute_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<boost::shared_ptr<std::vector<hpx::naming::id_type> > >::set_result_action,
    adaptive1d_dataflow_set_result_action_gid_vector_ptr);


#endif
