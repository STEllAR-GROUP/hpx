////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/runtime/agas/namespaces/factory.hpp>

typedef hpx::components::agas::server::factory_namespace_type<
    hpx::agas::tag::protocol_independent
>::type agas_component;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY
    (hpx::components::simple_component<agas_component>,
     factory_namespace);

HPX_DEFINE_GET_COMPONENT_TYPE(agas_component);

HPX_REGISTER_ACTION_EX
    (agas_component::bind_action, factory_namespace_bind_action);
HPX_REGISTER_ACTION_EX
    (agas_component::resolve_action, factory_namespace_resolve_action);
HPX_REGISTER_ACTION_EX
    (agas_component::unbind_action, factory_namespace_unbind_action);

