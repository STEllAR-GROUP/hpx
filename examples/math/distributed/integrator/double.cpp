////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <boost/serialization/vector.hpp>

#include <hpx/lcos/base_lco.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/util/serialize_rational.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <examples/math/distributed/integrator/server/integrator.hpp>

HPX_REGISTER_COMPONENT_MODULE();

///////////////////////////////////////////////////////////////////////////////
typedef hpx::balancing::server::integrator<double> integrator_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<
        integrator_type
    >, 
    integrator_double_factory);

HPX_REGISTER_ACTION_EX(
    integrator_type::build_network_action,
    integrator_double_build_network_action);

HPX_REGISTER_ACTION_EX(
    integrator_type::deploy_action,
    integrator_double_deploy_action);

HPX_REGISTER_ACTION_EX(
    integrator_type::solve_iterations_action,
    integrator_double_solve_iterations_action);

HPX_REGISTER_ACTION_EX(
    integrator_type::solve_action,
    integrator_double_solve_action);

HPX_DEFINE_GET_COMPONENT_TYPE(integrator_type);

