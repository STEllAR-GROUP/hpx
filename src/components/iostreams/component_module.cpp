//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/components/iostreams/server/output_stream.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

HPX_REGISTER_COMPONENT_MODULE()

typedef hpx::iostreams::server::output_stream ostream_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_EX(
    hpx::components::managed_component<ostream_type>,
    output_stream_factory, hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE(ostream_type)

HPX_REGISTER_ACTION(
    ostream_type::write_async_action,
    output_stream_write_async_action)

HPX_REGISTER_ACTION(
    ostream_type::write_sync_action,
    output_stream_write_sync_action)

