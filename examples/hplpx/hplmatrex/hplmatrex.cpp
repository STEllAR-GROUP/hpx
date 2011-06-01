////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2011 Dan Kogler
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include "server/hplmatrex.hpp"

///////////////////////////////////////////////////////////////
HPX_REGISTER_COMPONENT_MODULE();

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<hpx::components::server::HPLMatreX>,
    HPLMatreX);

//Register the actions
HPX_REGISTER_ACTION_EX(
    hpx::components::server::HPLMatreX::construct_action,HPLconstruct_action);
HPX_REGISTER_ACTION_EX(
    hpx::components::server::HPLMatreX::destruct_action,HPLdestruct_action);
HPX_REGISTER_ACTION_EX(
    hpx::components::server::HPLMatreX::assign_action,HPLassign_action);
HPX_REGISTER_ACTION_EX(
    hpx::components::server::HPLMatreX::solve_action,HPLsolve_action);
HPX_REGISTER_ACTION_EX(
    hpx::components::server::HPLMatreX::swap_action,HPLswap_action);
HPX_REGISTER_ACTION_EX(
    hpx::components::server::HPLMatreX::gmain_action,HPLgmain_action);
HPX_REGISTER_ACTION_EX(
    hpx::components::server::HPLMatreX::partbsub_action,HPLpartbsub_action);
HPX_REGISTER_ACTION_EX(
    hpx::components::server::HPLMatreX::check_action,HPLcheck_action);

HPX_DEFINE_GET_COMPONENT_TYPE(hpx::components::server::HPLMatreX);
