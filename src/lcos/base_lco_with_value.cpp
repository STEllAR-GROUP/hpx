//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/components/derived_component_factory_one.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/util/ini.hpp>
#include <hpx/util/serialize_exception.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/export.hpp>


namespace hpx { namespace lcos
{
}}

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_ACTION(
    hpx::lcos::base_lco_with_value<hpx::naming::gid_type>::set_value_action,
    set_value_action_gid_type)
HPX_REGISTER_ACTION(
    hpx::lcos::base_lco_with_value<hpx::naming::gid_type>::get_value_action,
    get_value_action_gid_type)
HPX_REGISTER_ACTION(
    hpx::lcos::base_lco_with_value<std::vector<hpx::naming::gid_type> >::set_value_action,
    set_value_action_vector_gid_type)
HPX_REGISTER_ACTION(
    hpx::lcos::base_lco_with_value<std::vector<hpx::naming::gid_type> >::get_value_action,
    get_value_action_vector_gid_type)
HPX_REGISTER_ACTION(
    hpx::lcos::base_lco_with_value<hpx::naming::id_type>::set_value_action,
    set_value_action_id_type)
HPX_REGISTER_ACTION(
    hpx::lcos::base_lco_with_value<hpx::naming::id_type>::get_value_action,
    get_value_action_id_type)
HPX_REGISTER_ACTION(
    hpx::lcos::base_lco_with_value<std::vector<hpx::naming::id_type> >::set_value_action,
    set_value_action_vector_id_type)
HPX_REGISTER_ACTION(
    hpx::lcos::base_lco_with_value<std::vector<hpx::naming::id_type> >::get_value_action,
    get_value_action_vector_id_type)
HPX_REGISTER_ACTION(
    hpx::lcos::base_lco_with_value<double>::set_value_action,
    set_value_action_double)
HPX_REGISTER_ACTION(
    hpx::lcos::base_lco_with_value<double>::get_value_action,
    get_value_action_double)
HPX_REGISTER_ACTION(
    hpx::lcos::base_lco_with_value<int>::set_value_action,
    set_value_action_int)
HPX_REGISTER_ACTION(
    hpx::lcos::base_lco_with_value<int>::get_value_action,
    get_value_action_int)
HPX_REGISTER_ACTION(
    hpx::lcos::base_lco_with_value<bool>::set_value_action,
    set_value_action_bool)
HPX_REGISTER_ACTION(
    hpx::lcos::base_lco_with_value<bool>::get_value_action,
    get_value_action_bool)
HPX_REGISTER_ACTION(
    hpx::lcos::base_lco_with_value<hpx::util::section>::set_value_action,
    set_value_action_section)
HPX_REGISTER_ACTION(
    hpx::lcos::base_lco_with_value<hpx::util::section>::get_value_action,
    get_value_action_section)
HPX_REGISTER_ACTION(
    hpx::lcos::base_lco_with_value<hpx::util::unused_type>::set_value_action,
    set_value_action_void)
HPX_REGISTER_ACTION(
    hpx::lcos::base_lco_with_value<hpx::util::unused_type>::get_value_action,
    get_value_action_void)
