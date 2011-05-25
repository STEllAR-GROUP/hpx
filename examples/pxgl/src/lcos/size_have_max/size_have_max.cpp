// Copyright (c) 2010-2011 Dylan Stark
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying 
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>

// Bring in necessary headers for setting up an HPX component
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

// Bring in PXGL headers
#include "../../../pxgl/util/futures.hpp"

#include "../../../pxgl/lcos/have_max.hpp"

////////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
HPX_REGISTER_COMPONENT_MODULE();

////////////////////////////////////////////////////////////////////////////////
// Define item types
typedef hpx::naming::id_type id_type;

//typedef int size_type;
typedef unsigned long size_type;

////////////////////////////////////////////////////////////////////////////////
// Define have-max type
typedef pxgl::lcos::server::have_max<size_type> size_have_max_type;

////////////////////////////////////////////////////////////////////////////////
// Register component factory for vectors
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::managed_component<size_have_max_type>,
    size_have_max);

////////////////////////////////////////////////////////////////////////////////
// Add serialization support for block vector actions
HPX_REGISTER_ACTION_EX(
    size_have_max_type::construct_action,
    size_have_max_construct_action);
HPX_REGISTER_ACTION_EX(
    size_have_max_type::signal_action,
    size_have_max_signal_action);

////////////////////////////////////////////////////////////////////////////////
// Define block vector component
HPX_DEFINE_GET_COMPONENT_TYPE(size_have_max_type);

////////////////////////////////////////////////////////////////////////////////
// Add futures support
HPX_REGISTER_FUTURE(bool, bool);
HPX_REGISTER_FUTURE(size_type, size);

