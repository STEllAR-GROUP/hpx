//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2009 Matt Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_DATAFLOW_SERVER_DZNAMIC_STENCIL_VALUE)
#define HPX_COMPONENTS_DATAFLOW_SERVER_DZNAMIC_STENCIL_VALUE

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/lcos/local/counting_semaphore.hpp>
#include <hpx/lcos/local/mutex.hpp>

#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/bind.hpp>

#include "stencil_value_in_adaptor.hpp"
#include "stencil_value_out_adaptor.hpp"
#include "../../parameter.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace adaptive1d { namespace server
{
    /// \class dynamic_stencil_value dynamic_stencil_value.hpp hpx/components/adaptive1d/server/dynamic_stencil_value.hpp
    class HPX_COMPONENT_EXPORT dynamic_stencil_value
      : public components::detail::managed_component_base<dynamic_stencil_value >
    {
    protected:
        // the in_adaptors_type is the concrete dynamic_stencil_value_in_adaptor
        // of the proper type
        typedef adaptive1d::server::stencil_value_in_adaptor in_adaptor_type;

        // the out_adaptors_type is the concrete dynamic_stencil_value_out_adaptor
        // of the proper type
        typedef
            managed_component<adaptive1d::server::stencil_value_out_adaptor>
        out_adaptor_type;

    public:
        /// Construct a new dynamic_stencil_value instance
        dynamic_stencil_value();

        /// Destruct this stencil instance
        ~dynamic_stencil_value();

        /// \brief finalize() will be called just before the instance gets
        ///        destructed
        ///
        /// \param appl [in] The applier to be used for finalization of the
        ///             component instance.
        void finalize();

        /// The function get will be called by the out-ports whenever
        /// the current value has been requested.
        naming::id_type get_value(int i);

        ///////////////////////////////////////////////////////////////////////
        // parcel action code: the action to be performed on the destination
        // object (the accumulator)
        enum actions
        {
            dynamic_stencil_value_call = 0,
            dynamic_stencil_value_get_output_ports = 1,
            dynamic_stencil_value_connect_input_ports = 2,
            dynamic_stencil_value_set_functional_component = 3,
            dynamic_stencil_value_start = 4,
        };

        /// Main thread function looping through all timesteps
        void main();

        /// This is the main entry point of this component. Calling this
        /// function (by applying the call_action) will trigger the repeated
        /// execution of the whole time step evolution functionality.
        ///
        /// It invokes the time series evolution for this data point using the
        /// data referred to by the parameter \a initial. After finishing
        /// execution it returns a reference to the result as its return value
        /// (parameter \a result)
        naming::id_type call(naming::id_type const& initial);

        /// Return the gid's of the output ports associated with this
        /// \a dynamic_stencil_value instance.
        std::vector<naming::id_type> get_output_ports();

        /// Connect the destinations given by the provided gid's with the
        /// corresponding input ports associated with this \a dynamic_stencil_value
        /// instance.
        util::unused_type
        connect_input_ports(std::vector<naming::id_type> const& gids);

        /// Set the gid of the component implementing the actual time evolution
        /// functionality
        util::unused_type
        set_functional_component(naming::id_type const& gid, int row,
            int column, int instencilsize, int outstencilsize,
            double cycle_time,parameter const& par);

        util::unused_type start();

        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::result_action1<
            dynamic_stencil_value, naming::id_type, dynamic_stencil_value_call,
            naming::id_type const&, &dynamic_stencil_value::call
        > call_action;

        typedef hpx::actions::result_action0<
            dynamic_stencil_value, std::vector<naming::id_type>,
            dynamic_stencil_value_get_output_ports,
            &dynamic_stencil_value::get_output_ports
        > get_output_ports_action;

        typedef hpx::actions::result_action1<
            dynamic_stencil_value, util::unused_type,
            dynamic_stencil_value_connect_input_ports,
            std::vector<naming::id_type> const&,
            &dynamic_stencil_value::connect_input_ports
        > connect_input_ports_action;

        typedef hpx::actions::result_action7<
            dynamic_stencil_value, util::unused_type,
            dynamic_stencil_value_set_functional_component,
            naming::id_type const&, int, int, int, int,double, parameter const&,
            &dynamic_stencil_value::set_functional_component
        > set_functional_component_action;

        typedef hpx::actions::result_action0<
            dynamic_stencil_value, util::unused_type,
            dynamic_stencil_value_start, &dynamic_stencil_value::start
        > start_action;

    private:
        bool is_called_;                              // is one of the 'main' stencils
        threads::thread_id_type driver_thread_;

        std::vector<boost::shared_ptr<lcos::local::counting_semaphore> > sem_in_;
        std::vector<boost::shared_ptr<lcos::local::counting_semaphore> > sem_out_;
        lcos::local::counting_semaphore sem_result_;

        std::vector<boost::shared_ptr<in_adaptor_type> > in_;   // adaptors used to gather input
        std::vector<naming::id_type> out_;                      // adaptors used to provide result

        naming::id_type value_gids_[2];               // reference to previous values
        naming::id_type functional_gid_;              // reference to functional code

        int row_;             // position of this stencil in whole graph
        int column_;
        std::size_t instencilsize_;
        std::size_t outstencilsize_;
        parameter par_;
        double cycle_time_;

        typedef lcos::local::mutex mutex_type;
        mutex_type mtx_;
    };

}}}}

HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::adaptive1d::server::dynamic_stencil_value::call_action,
    adaptive1d_dataflow_dynamic_stencil_value_double_call_action);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::adaptive1d::server::dynamic_stencil_value::get_output_ports_action,
    adaptive1d_dataflow_dynamic_stencil_value_double_get_output_ports_action);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::adaptive1d::server::dynamic_stencil_value::connect_input_ports_action,
    adaptive1d_dataflow_dynamic_stencil_value_double_connect_input_ports_action);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::adaptive1d::server::dynamic_stencil_value::set_functional_component_action,
    adaptive1d_dataflow_dynamic_stencil_value_double_set_functional_component_action);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::adaptive1d::server::dynamic_stencil_value::start_action,
    adaptive1d_dataflow_dynamic_stencil_value_double_start_action);

#endif

