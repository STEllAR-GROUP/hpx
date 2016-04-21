//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/components/derived_component_factory.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/lcos/base_lco_with_value.hpp>
#include <hpx/util/ini.hpp>
#include <hpx/util/serialize_exception.hpp>

HPX_REGISTER_BASE_LCO_WITH_VALUE_ID(
    boost::int16_t, int16_t,
    hpx::actions::base_lco_with_value_int16_get,
    hpx::actions::base_lco_with_value_int16_set)
HPX_REGISTER_BASE_LCO_WITH_VALUE_ID(
    boost::uint16_t, uint16_t,
    hpx::actions::base_lco_with_value_uint16_get,
    hpx::actions::base_lco_with_value_uint16_set)
HPX_REGISTER_BASE_LCO_WITH_VALUE_ID(
    boost::int32_t, int32_t,
    hpx::actions::base_lco_with_value_int32_get,
    hpx::actions::base_lco_with_value_int32_set)
HPX_REGISTER_BASE_LCO_WITH_VALUE_ID(
    boost::uint32_t, uint32_t,
    hpx::actions::base_lco_with_value_uint32_get,
    hpx::actions::base_lco_with_value_uint32_set)
