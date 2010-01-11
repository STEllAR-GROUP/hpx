//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/lcos/future_wait.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>
#include <boost/assign/std/vector.hpp>

#include "../dynamic_stencil_value.hpp"
#include "../functional_component.hpp"
#include "../../parameter.hpp"

#include "tapered_autonomous.hpp"

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::amr::server::tapered_autonomous had_tapered_autonomous_type;

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the actions
HPX_REGISTER_ACTION_EX(had_tapered_autonomous_type::init_execute_action, had_tapered_autonomous_init_execute_action);
HPX_REGISTER_ACTION_EX(had_tapered_autonomous_type::execute_action, had_tapered_autonomous_execute_action);

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<had_tapered_autonomous_type>, had_tapered_autonomous);
HPX_DEFINE_GET_COMPONENT_TYPE(had_tapered_autonomous_type);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace server 
{
    tapered_autonomous::tapered_autonomous()
      : numvalues_(0)
    {}

    ///////////////////////////////////////////////////////////////////////////////
    // do actual work
    void tapered_autonomous::execute(
        components::distributing_factory::iterator_range_type const& stencils, 
        std::vector<double> const& initial_data, 
        std::vector<naming::id_type>& result_data)
    {
    }
    
    ///////////////////////////////////////////////////////////////////////////
    /// This is the main entry point of this component. 
    std::vector<naming::id_type> tapered_autonomous::init_execute(
        components::component_type function_type,
        components::component_type logging_type, 
        std::size_t level, double x, Parameter const& par)
    {
        std::size_t numvalues = 8;
        std::size_t numsteps = 2;

        std::vector<naming::id_type> result_data;

        return result_data;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// This the other entry point of this component. 
    std::vector<naming::id_type> tapered_autonomous::execute(
        std::vector<double> const& initial_data,
        components::component_type function_type,
        components::component_type logging_type, Parameter const& par)
    {
        std::size_t numvalues = 8;
        std::size_t numsteps = 2;

        std::vector<naming::id_type> result_data;

        return result_data;
    }

}}}}
