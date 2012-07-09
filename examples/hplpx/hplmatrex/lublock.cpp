////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2011 Dan Kogler
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/vector.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include "lublock.hpp"

///////////////////////////////////////////////////////////////
HPX_REGISTER_COMPONENT_MODULE();

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<hpx::components::server::lublock>,
    lublock);

HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::lcos::base_lco_with_value<std::vector<std::vector<double> > >,
    hpx::components::component_base_lco_with_value);

HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<
        std::vector<std::vector<double> >
    >::set_value_action,
    base_lco_with_value_set_value_vector_vector_double);

//Register the actions
HPX_REGISTER_ACTION_EX(
    hpx::components::server::lublock::constructBlock_action,HPLconstructBlock_action);
HPX_REGISTER_ACTION_EX(
    hpx::components::server::lublock::gcorner_action,HPLgcorner_action);
HPX_REGISTER_ACTION_EX(
    hpx::components::server::lublock::gtop_action,HPLgtop_action);
HPX_REGISTER_ACTION_EX(
    hpx::components::server::lublock::gleft_action,HPLgleft_action);
HPX_REGISTER_ACTION_EX(
    hpx::components::server::lublock::gtrail_action,HPLgtrail_action);
HPX_REGISTER_ACTION_EX(
    hpx::components::server::lublock::getRows_action,HPLgetRows_action);
HPX_REGISTER_ACTION_EX(
    hpx::components::server::lublock::getColumns_action,HPLgetColumns_action);
HPX_REGISTER_ACTION_EX(
    hpx::components::server::lublock::getData_action,HPLgetData_action);
HPX_REGISTER_ACTION_EX(
    hpx::components::server::lublock::getFuture_action,HPLgetFuture_action);
