////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_D9951196_521D_4EA5_947D_43451437AEE6)
#define HPX_D9951196_521D_4EA5_947D_43451437AEE6

#include <multimap>

#include <hpx/runtime/agas/traits.hpp>
#include <hpx/runtime/agas/basic_namespace.hpp>

namespace hpx { namespace agas // hpx::agas
{

namespace tag { // hpx::agas::tag

struct factory_namespace;

} // hpx::agas::tag

namespace traits { // hpx::agas::traits

template <>
struct registry_type<tag::factory_namespace>
{ typedef std::multimap<naming::gid_type, boost::uint32_t> type; };

// TODO: implement bind_hook, update_hook, resolve_hook and unbind_hook

} // hpx::agas::traits
} // hpx::agas

///////////////////////////////////////////////////////////////////////////////
namespace components { namespace agas // hpx::components::agas
{

struct factory_namespace
  : private basic_namespace<hpx::agas::tag::factory_namespace>
{
    // TODO: implement interface
};

// MPL metafunction (syntactic sugar)
template <typename Protocal>
struct factory_namespace_type
{ typedef factory_namespace type; }; 

///////////////////////////////////////////////////////////////////////////////
namespace server // hpx::components::agas::server
{

// MPL metafunction
template <typename Protocal>
struct factory_namespace_type
{ typedef server::basic_namespace<hpx::agas::tag::factory_namespace> type };

} // hpx::components::agas::server

///////////////////////////////////////////////////////////////////////////////
namespace stubs // hpx::components::agas::stubs
{

// MPL metafunction
template <typename Protocal>
struct factory_namespace_type
{ typedef stubs::basic_namespace<hpx::agas::tag::factory_namespace> type };

} // hpx::components::agas::stubs

} // hpx::components::agas
} // hpx::components
} // hpx

#endif // HPX_D9951196_521D_4EA5_947D_43451437AEE6

