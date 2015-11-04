//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/components/derived_component_factory.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/lcos/base_lco_with_value.hpp>
#include <hpx/util/ini.hpp>
#include <hpx/util/serialize_exception.hpp>

HPX_REGISTER_BASE_LCO_WITH_VALUE_ID(
    float, float,
    hpx::actions::base_lco_with_value_float_get,
    hpx::actions::base_lco_with_value_float_set)
HPX_REGISTER_BASE_LCO_WITH_VALUE_ID(
    double, double,
    hpx::actions::base_lco_with_value_double_get,
    hpx::actions::base_lco_with_value_double_set)
HPX_REGISTER_BASE_LCO_WITH_VALUE_ID(
    boost::int8_t, int8_t,
    hpx::actions::base_lco_with_value_int8_get,
    hpx::actions::base_lco_with_value_int8_set)
HPX_REGISTER_BASE_LCO_WITH_VALUE_ID(
    boost::uint8_t, uint8_t,
    hpx::actions::base_lco_with_value_uint8_get,
    hpx::actions::base_lco_with_value_uint8_set)
