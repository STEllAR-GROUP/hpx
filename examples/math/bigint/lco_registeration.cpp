////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <boost/bigint.hpp>
#include <boost/bigint/serialize.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/export.hpp>

#include <hpx/hpx.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

using boost::bigint;

using hpx::components::component_base_lco_with_value;

using hpx::lcos::base_lco_with_value;

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_COMPONENT_MODULE();

HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    base_lco_with_value<bigint>,
    component_base_lco_with_value);

HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<
        boost::bigint
    >::set_result_action,
    base_lco_with_value_set_result_bigint);

HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    base_lco_with_value<boost::shared_ptr<bigint> >,
    component_base_lco_with_value);

HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<
        boost::shared_ptr<boost::bigint>
    >::set_result_action,
    base_lco_with_value_set_result_bigint_shared_ptr);

