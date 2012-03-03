////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/vector.hpp>

#include <vector>

#include <hpx/hpx.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/agas/response.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

using hpx::lcos::base_lco_with_value;

using hpx::components::component_base_lco_with_value;

using hpx::agas::response;

using hpx::naming::id_type;

HPX_REGISTER_ACTION_EX(
    base_lco_with_value<response>::set_result_action,
    set_result_action_agas_response_type);

typedef base_lco_with_value<bool, response> base_lco_bool_response_type;
HPX_REGISTER_ACTION_EX(
    base_lco_bool_response_type::set_result_action,
    set_result_action_agas_bool_response_type);

typedef base_lco_with_value<id_type, response> base_lco_id_type_response_type;
HPX_REGISTER_ACTION_EX(
    base_lco_id_type_response_type::set_result_action,
    set_result_action_agas_id_type_response_type);

typedef base_lco_with_value<std::vector<response> >
    base_lco_vector_response_type;
HPX_REGISTER_ACTION_EX(
    base_lco_vector_response_type::set_result_action,
    set_result_action_agas_vector_response_type);

