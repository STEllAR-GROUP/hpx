//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/util/ini.hpp>
#include <hpx/util/serialize_exception.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/export.hpp>

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the future actions
HPX_REGISTER_ACTION_EX(hpx::lcos::base_lco::set_event_action, base_set_event_action);
HPX_REGISTER_ACTION_EX(hpx::lcos::base_lco::set_error_action, base_set_error_action);

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<hpx::naming::id_type>::set_result_action, 
    set_result_action_id_type);
HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<std::vector<hpx::naming::id_type> >::set_result_action,
    set_result_action_vector_id_type);
HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<double>::set_result_action,
    set_result_action_double);
HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<int>::set_result_action,
    set_result_action_int);
HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<hpx::util::section>::set_result_action,
    set_result_action_section);

///////////////////////////////////////////////////////////////////////////////
HPX_DEFINE_GET_COMPONENT_TYPE(hpx::lcos::base_lco);
HPX_DEFINE_GET_COMPONENT_TYPE(hpx::lcos::base_lco_with_value<hpx::naming::id_type>);
HPX_DEFINE_GET_COMPONENT_TYPE(hpx::lcos::base_lco_with_value<std::vector<hpx::naming::id_type> >);
HPX_DEFINE_GET_COMPONENT_TYPE(hpx::lcos::base_lco_with_value<double>);
HPX_DEFINE_GET_COMPONENT_TYPE(hpx::lcos::base_lco_with_value<int>);
HPX_DEFINE_GET_COMPONENT_TYPE(hpx::lcos::base_lco_with_value<hpx::util::section>);

//////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    template hpx::util::section const& 
    continuation::trigger(hpx::util::section const&);
}}

