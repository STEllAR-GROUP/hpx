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
    hpx::components::simple_component<hpx::components::server::hplmatrex>,
    hplmatrex);

//Register the actions
HPX_REGISTER_ACTION_EX(
    hpx::components::server::hplmatrex::construct_action,HPLconstruct_action);
HPX_REGISTER_ACTION_EX(
    hpx::components::server::hplmatrex::assign_action,HPLassign_action);
HPX_REGISTER_ACTION_EX(
    hpx::components::server::hplmatrex::solve_action,HPLsolve_action);
HPX_REGISTER_ACTION_EX(
    hpx::components::server::hplmatrex::search_action,HPLsearch_action);
HPX_REGISTER_ACTION_EX(
    hpx::components::server::hplmatrex::swap_action,HPLswap_action);
HPX_REGISTER_ACTION_EX(
    hpx::components::server::hplmatrex::partbsub_action,HPLpartbsub_action);
HPX_REGISTER_ACTION_EX(
    hpx::components::server::hplmatrex::check_action,HPLcheck_action);

