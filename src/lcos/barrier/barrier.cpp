//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/derived_component_factory_one.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/lcos/server/barrier.hpp>
#include <hpx/util/serialize_exception.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

///////////////////////////////////////////////////////////////////////////////
// Barrier
typedef hpx::components::managed_component<hpx::lcos::server::barrier> barrier_type;

HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::lcos::server::barrier, hpx::components::component_barrier)
HPX_REGISTER_DERIVED_COMPONENT_FACTORY_ONE(barrier_type, barrier,
    "hpx::lcos::base_lco", hpx::components::factory_enabled)

