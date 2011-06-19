//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/derived_component_factory.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include <hpx/lcos/server/object_semaphore.hpp>

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
HPX_REGISTER_COMPONENT_MODULE();

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::managed_component<
    hpx::lcos::server::object_semaphore<int>
> int_object_semaphore_type;

typedef hpx::lcos::detail::local_dataflow_variable<int, int> int_dataflow_type;
HPX_DEFINE_GET_COMPONENT_TYPE(int_dataflow_type);

HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    int_object_semaphore_type::base_type_holder,
    hpx::components::component_base_lco_with_value);

HPX_REGISTER_ACTION_EX(
    hpx::lcos::server::object_semaphore<int>::set_result_action,
    int_object_semaphore_set_result);

HPX_REGISTER_ACTION_EX(
    hpx::lcos::server::object_semaphore<int>::get_value_action,
    int_object_semaphore_get_value);

HPX_REGISTER_ACTION_EX(
    hpx::lcos::server::object_semaphore<int>::add_lco_action,
    int_object_semaphore_add_lco);

HPX_REGISTER_DERIVED_COMPONENT_FACTORY(
    int_object_semaphore_type
  , int_object_semaphore
  , "hpx::lcos::base_lco_with_value<"
        "boost::fusion::vector2<int, boost::uint64_t>"
      ", boost::fusion::vector2<int, boost::uint64_t> "
    ">");

HPX_DEFINE_GET_COMPONENT_TYPE(int_object_semaphore_type::wrapped_type);

