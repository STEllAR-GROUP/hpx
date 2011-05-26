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

#include <hpx/runtime/agas/database/backend/stdmap.hpp>

using hpx::components::component_agas_symbol_namespace;

#if !defined(HPX_AGAS_SYSTEM)
    #include <hpx/runtime/agas/namespace/user_symbol.hpp>

    typedef hpx::agas::server::user_symbol_namespace<
        hpx::agas::tag::database::stdmap
    > agas_component;
    
    HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
        hpx::components::simple_component<agas_component>,
        stdmap_symbol_namespace);
    
    HPX_DEFINE_GET_COMPONENT_TYPE(agas_component::base_type);
    
    namespace hpx { namespace components
    {
    
    template <> HPX_ALWAYS_EXPORT
    component_type component_type_database<agas_component>::get()
    { return component_type_database<agas_component::base_type>::get(); }
    
    template <> HPX_ALWAYS_EXPORT
    void component_type_database<agas_component>::set(component_type t)
    { component_type_database<agas_component::base_type>::set(t); }
    
    }}
#else    
    #include <hpx/runtime/agas/namespace/bootstrap_symbol.hpp>

    typedef hpx::agas::server::bootstrap_symbol_namespace<
        hpx::agas::tag::database::stdmap
    > agas_component;
    
    HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
        hpx::components::fixed_component<agas_component>,
        stdmap_bootstrap_symbol_namespace);

    HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
        agas_component, component_agas_symbol_namespace);
    HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
        agas_component::base_type, component_agas_symbol_namespace);
#endif

HPX_REGISTER_ACTION_EX(
    agas_component::bind_action,
    stdmap_symbol_namespace_bind_action);
HPX_REGISTER_ACTION_EX(
    agas_component::rebind_action,
    stdmap_symbol_namespace_rebind_action);
HPX_REGISTER_ACTION_EX(
    agas_component::resolve_action,
    stdmap_symbol_namespace_resolve_action);
HPX_REGISTER_ACTION_EX(
    agas_component::unbind_action,
    stdmap_symbol_namespace_unbind_action);

