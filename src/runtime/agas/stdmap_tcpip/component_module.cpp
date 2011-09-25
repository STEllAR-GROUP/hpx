////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/version.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/optional.hpp>

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/lcos/base_lco.hpp>

#include <hpx/runtime/agas/network/backend/tcpip.hpp>
#include <hpx/runtime/agas/namespace/response.hpp>

#include <hpx/runtime/components/component_factory.hpp>

using hpx::lcos::base_lco_with_value;
using hpx::components::component_base_lco_with_value;

typedef hpx::agas::response< 
    hpx::agas::tag::network::tcpip
> response_type;

HPX_REGISTER_ACTION_EX(
    base_lco_with_value<response_type>::set_result_action,
    set_result_action_agas_stdmap_tcpip_response_type);
HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    base_lco_with_value<response_type>,
    component_base_lco_with_value);

