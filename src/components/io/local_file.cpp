//  Copyright (c) 2014 Shuangyang Yang
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/serialization.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <hpx/components/io/local_file.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
HPX_REGISTER_COMPONENT_MODULE()

///////////////////////////////////////////////////////////////////////////////
typedef hpx::io::server::local_file local_file_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::managed_component<local_file_type>,
    local_file, hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE(local_file_type)

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the local_file actions
HPX_REGISTER_ACTION(
    local_file_type::open_action,
    local_file_open_action)
HPX_REGISTER_ACTION(
    local_file_type::is_open_action,
    local_file_is_open_action)
HPX_REGISTER_ACTION(
    local_file_type::close_action,
    local_file_close_action)
HPX_REGISTER_ACTION(
    local_file_type::remove_file_action,
    local_file_remove_file_action)
HPX_REGISTER_ACTION(
    local_file_type::read_action,
    local_file_read_action)
HPX_REGISTER_ACTION(
    local_file_type::pread_action,
    local_file_pread_action)
HPX_REGISTER_ACTION(
    local_file_type::write_action,
    local_file_write_action)
HPX_REGISTER_ACTION(
    local_file_type::pwrite_action,
    local_file_pwrite_action)
HPX_REGISTER_ACTION(
    local_file_type::lseek_action,
    local_file_lseek_action)
