//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>
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

HPX_DEFINE_GET_COMPONENT_TYPE(int_object_semaphore_type::wrapped_type);

HPX_REGISTER_ACTION_EX(
    hpx::lcos::server::object_semaphore<int>::signal_action,
    int_object_semaphore_signal);

HPX_REGISTER_ACTION_EX(
    hpx::lcos::server::object_semaphore<int>::get_action,
    int_object_semaphore_get);

HPX_REGISTER_ACTION_EX(
    hpx::lcos::server::object_semaphore<int>::abort_pending_action,
    int_object_semaphore_abort_pending);

HPX_REGISTER_ACTION_EX(
    hpx::lcos::server::object_semaphore<int>::wait_action,
    int_object_semaphore_wait);

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::managed_component<
        hpx::lcos::server::object_semaphore<int>
    >,
    int_object_semaphore);

