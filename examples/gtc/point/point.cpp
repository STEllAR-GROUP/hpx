//  Copyright (c) 2007-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include "server/point.hpp"

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality.
HPX_REGISTER_COMPONENT_MODULE();

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::managed_component<
    gtc::server::point
> gtc_point_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(gtc_point_type, gtc_point);

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_ACTION_EX(
    gtc_point_type::wrapped_type::init_action,
    gtc_point_init_action);

HPX_REGISTER_ACTION_EX(
    gtc_point_type::wrapped_type::load_action,
    gtc_point_load_action);

HPX_REGISTER_ACTION_EX(
    gtc_point_type::wrapped_type::chargei_action,
    gtc_point_chargei_action);

HPX_REGISTER_ACTION_EX(
    gtc_point_type::wrapped_type::get_densityi_action,
    gtc_point_get_densityi_action);

HPX_REGISTER_ACTION_EX(
    gtc_point_type::wrapped_type::get_zonali_action,
    gtc_point_get_zonali_action);

HPX_REGISTER_ACTION_EX(
    gtc_point_type::wrapped_type::smooth_action,
    gtc_point_smooth_action);

HPX_REGISTER_ACTION_EX(
    gtc_point_type::wrapped_type::get_phi_action,
    gtc_point_get_phi_action);

HPX_REGISTER_ACTION_EX(
    gtc_point_type::wrapped_type::get_eachzeta_action,
    gtc_point_get_eachzeta_action);

HPX_REGISTER_ACTION_EX(
    gtc_point_type::wrapped_type::field_action,
    gtc_point_field_action);

HPX_REGISTER_ACTION_EX(
    gtc_point_type::wrapped_type::get_evector_action,
    gtc_point_get_evector_action);

HPX_REGISTER_ACTION_EX(
    gtc_point_type::wrapped_type::pushi_action,
    gtc_point_pushi_action);

HPX_REGISTER_ACTION_EX(
    gtc_point_type::wrapped_type::get_dden_action,
    gtc_point_get_dden_action);

HPX_REGISTER_ACTION_EX(
    gtc_point_type::wrapped_type::get_dtem_action,
    gtc_point_get_dtem_action);

HPX_REGISTER_ACTION_EX(
    gtc_point_type::wrapped_type::shifti_action,
    gtc_point_shifti_action);

HPX_REGISTER_ACTION_EX(
    gtc_point_type::wrapped_type::get_msend_action,
    gtc_point_get_msend_action);

HPX_REGISTER_ACTION_EX(
    gtc_point_type::wrapped_type::get_msendright_action,
    gtc_point_get_msendright_action);

HPX_REGISTER_ACTION_EX(
    gtc_point_type::wrapped_type::get_sendright_action,
    gtc_point_get_sendright_action);

HPX_REGISTER_ACTION_EX(
    gtc_point_type::wrapped_type::get_msendleft_action,
    gtc_point_get_msendleft_action);

HPX_REGISTER_ACTION_EX(
    gtc_point_type::wrapped_type::get_sendleft_action,
    gtc_point_get_sendleft_action);

HPX_REGISTER_ACTION_EX(
    gtc_point_type::wrapped_type::poisson_action,
    gtc_point_poisson_action);
