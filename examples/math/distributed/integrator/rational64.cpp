////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <boost/rational.hpp>
#include <boost/serialization/vector.hpp>

#include <hpx/lcos/base_lco.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/util/serialize_rational.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <examples/math/distributed/integrator/server/integrator.hpp>

///////////////////////////////////////////////////////////////////////////////
typedef boost::rational<boost::int64_t> rational64;

HPX_REGISTER_COMPONENT_MODULE();

HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::lcos::base_lco_with_value<rational64>,
    component_base_lco_with_value);

HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<
        boost::rational<boost::int64_t>
    >::set_result_action,
    base_lco_with_value_set_result_rational64);

///////////////////////////////////////////////////////////////////////////////
typedef hpx::balancing::server::integrator<rational64> integrator_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<
        integrator_type
    >, 
    integrator_rational64_factory);

HPX_REGISTER_ACTION_EX(
    integrator_type::build_network_action,
    integrator_rational64_build_network_action);

HPX_REGISTER_ACTION_EX(
    integrator_type::deploy_action,
    integrator_rational64_deploy_action);

HPX_REGISTER_ACTION_EX(
    integrator_type::solve_iteration_action,
    integrator_rational64_solve_iteration_action);

HPX_REGISTER_ACTION_EX(
    integrator_type::solve_action,
    integrator_rational64_solve_action);

HPX_DEFINE_GET_COMPONENT_TYPE(integrator_type);

