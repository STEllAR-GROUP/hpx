//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/actions/continuation.hpp>

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

HPX_DEFINE_GET_COMPONENT_TYPE(int_object_semaphore_type::wrapped_type);

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::managed_component<
        hpx::lcos::server::object_semaphore<int>
    >,
    int_object_semaphore);

