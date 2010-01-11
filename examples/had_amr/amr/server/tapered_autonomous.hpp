//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_SERVER_TAPERED_AUTONOMOUS_JAN_11_2010_1131AM)
#define HPX_COMPONENTS_AMR_SERVER_TAPERED_AUTONOMOUS_JAN_11_2010_1131AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/components/distributing_factory/distributing_factory.hpp>

#include "../../parameter.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace server 
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT tapered_autonomous
      : public simple_component_base<tapered_autonomous>
    {
    private:
        typedef simple_component_base<tapered_autonomous> base_type;

    public:
        tapered_autonomous();

        // components must contain a typedef for wrapping_type defining the
        // component type used to encapsulate instances of this component
        typedef amr::server::tapered_autonomous wrapping_type;

        ///////////////////////////////////////////////////////////////////////
        // parcel action code: the action to be performed on the destination 
        // object (the accumulator)
        enum actions
        {
            tapered_autonomous_init_execute = 0,
            tapered_autonomous_execute = 1
        };

        /// This is the main entry point of this component. 
        std::vector<naming::id_type> init_execute(
            components::component_type function_type,
            components::component_type logging_type, 
            std::size_t level, double x, Parameter const& par);

        std::vector<naming::id_type> execute(
            std::vector<double> const& initialdata,
            components::component_type function_type,
            components::component_type logging_type, Parameter const& par);

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::result_action5<
            tapered_autonomous, std::vector<naming::id_type>, tapered_autonomous_init_execute, 
            components::component_type, 
            components::component_type, std::size_t, double,
            Parameter const&, &tapered_autonomous::init_execute
        > init_execute_action;

        typedef hpx::actions::result_action4<
            tapered_autonomous, std::vector<naming::id_type>, tapered_autonomous_execute, 
            std::vector<double> const&,
            components::component_type, 
            components::component_type, Parameter const&, &tapered_autonomous::execute
        > execute_action;

    protected:
        typedef 
            components::distributing_factory::iterator_range_type
        distributed_iterator_range_type;

        static void init(distributed_iterator_range_type const& functions,
            distributed_iterator_range_type const& logging,
            std::size_t numsteps);

        static void execute(distributed_iterator_range_type const& stencils, 
            std::vector<double> const& initial_data, 
            std::vector<naming::id_type>& result_data);

    private:
        std::size_t numvalues_;
    };

}}}}

#endif
