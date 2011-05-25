// Copyright (c) 2010-2011 Dylan Stark
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying 
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(PXGL_UTIL_COMPONENT_20101025T1041)
#define PXGL_UTIL_COMPONENT_20101025T1041

#include <hpx/hpx.hpp>
#include <hpx/hpx_fwd.hpp>

#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>

#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>

#include <hpx/runtime/components/client_base.hpp>

////////////////////////////////////////////////////////////////////////////////
#define HPX_MANAGED_BASE_0(name) hpx::components::detail::managed_component_base<name>

#define HPX_STUBS_BASE_0(name) hpx::components::stubs::stub_base<server::name>

#define HPX_CLIENT_BASE_0(name) hpx::components::client_base<name, stubs::name>

////////////////////////////////////////////////////////////////////////////////
#define HPX_MANAGED_BASE_1(name, T1) hpx::components::detail::managed_component_base<name<T1> >

#define HPX_STUBS_BASE_1(name, T1) hpx::components::stubs::stub_base<server::name<T1> >

#define HPX_CLIENT_BASE_1(name, T1) hpx::components::client_base<name<T1>, stubs::name<T1> >

////////////////////////////////////////////////////////////////////////////////
#define HPX_MANAGED_BASE_2(name, T1, T2) hpx::components::detail::managed_component_base<name<T1, T2> >

#define HPX_STUBS_BASE_2(name, T1, T2) hpx::components::stubs::stub_base<server::name<T1, T2> >

#define HPX_CLIENT_BASE_2(name, T1, T2) hpx::components::client_base<name<T1, T2>, stubs::name<T1, T2> >

#endif

