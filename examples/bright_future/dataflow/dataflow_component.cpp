
//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/derived_component_factory.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/spinlock.hpp>
#include <hpx/util/unlock_lock.hpp>
#include <hpx/util/stringstream.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/traits/get_remote_result.hpp>


#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>

#include <examples/bright_future/dataflow/server/dataflow.hpp>

HPX_REGISTER_COMPONENT_MODULE();

typedef hpx::components::managed_component<hpx::lcos::server::dataflow> dataflow_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_EX(
    dataflow_type,
    dataflow_factory, true);

HPX_DEFINE_GET_COMPONENT_TYPE(dataflow_type::wrapped_type);
