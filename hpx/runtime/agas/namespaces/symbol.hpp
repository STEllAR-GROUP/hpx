////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_544F0DE6_CC7A_4644_A6BA_2F89F9526DB5)
#define HPX_544F0DE6_CC7A_4644_A6BA_2F89F9526DB5

#include <string>
#include <map>

#include <hpx/runtime/agas/traits.hpp>
#include <hpx/runtime/agas/basic_namespace.hpp>

namespace hpx { namespace agas // hpx::agas
{

namespace tag { // hpx::agas::tag

struct symbol_namespace;

} // hpx::agas::tag

namespace traits { // hpx::agas::traits

template <>
struct registry_type<tag::symbol_namespace>
{ typedef std::map<std::string, naming::gid_type> type; };

// TODO: implement bind_hook, update_hook, resolve_hook and unbind_hook

} // hpx::agas::traits
} // hpx::agas

///////////////////////////////////////////////////////////////////////////////
namespace components { namespace agas // hpx::components::agas
{

struct symbol_namespace
  : private basic_namespace<hpx::agas::tag::symbol_namespace>
{
    // TODO: implement interface
};

// MPL metafunction (syntactic sugar)
template <typename Protocal>
struct symbol_namespace_type
{ typedef symbol_namespace type; }; 

///////////////////////////////////////////////////////////////////////////////
namespace server // hpx::components::agas::server
{

// MPL metafunction
template <typename Protocal>
struct symbol_namespace_type
{ typedef server::basic_namespace<hpx::agas::tag::symbol_namespace> type; };

} // hpx::components::agas::server

///////////////////////////////////////////////////////////////////////////////
namespace stubs // hpx::components::agas::stubs
{

// MPL metafunction
template <typename Protocal>
struct symbol_namespace_type
{ typedef stubs::basic_namespace<hpx::agas::tag::symbol_namespace> type; };

} // hpx::components::agas::stubs

} // hpx::components::agas
} // hpx::components
} // hpx

#endif // HPX_544F0DE6_CC7A_4644_A6BA_2F89F9526DB5

