////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_352F80CF_5A0C_4DAE_9DC3_8CEE5F789A75)
#define HPX_352F80CF_5A0C_4DAE_9DC3_8CEE5F789A75

#include <map>

#include <hpx/runtime/agas/traits.hpp>
#include <hpx/runtime/agas/partition.hpp>
#include <hpx/runtime/agas/basic_namespace.hpp>

namespace hpx { namespace agas // hpx::agas
{

namespace tag { // hpx::agas::tag

template <typename Protocal>
struct locality_namespace;

} // hpx::agas::tag

namespace traits { // hpx::agas::traits

template <typename Protocal>
struct registry_type<tag::locality_namespace<Protocal> >
{
    typedef std::map<typename locality_type<Protocal>::type, partition>
        type;
};

// TODO: implement bind_hook, update_hook, resolve_hook and unbind_hook

} // hpx::agas::traits
} // hpx::agas

///////////////////////////////////////////////////////////////////////////////
namespace components { namespace agas // hpx::components::agas
{

template <typename Protocal>
struct locality_namespace
  : private basic_namespace<hpx::agas::tag::locality_namespace<Protocal> >
{
    // TODO: implement interface
};

// MPL metafunction (syntactic sugar)
template <typename Protocal>
struct locality_namespace_type
{ typedef locality_namespace<Protocal> type; }; 

///////////////////////////////////////////////////////////////////////////////
namespace server // hpx::components::agas::server
{

// MPL metafunction
template <typename Protocal>
struct locality_namespace_type
{
    typedef server::basic_namespace<
        hpx::agas::tag::locality_namespace<Protocal>
    > type;
};

} // hpx::components::agas::server

///////////////////////////////////////////////////////////////////////////////
namespace stubs // hpx::components::agas::stubs
{

// MPL metafunction
template <typename Protocal>
struct locality_namespace_type
{
    typedef stubs::basic_namespace<
        hpx::agas::tag::locality_namespace<Protocal>
    > type;
};

} // hpx::components::agas::stubs

} // hpx::components::agas
} // hpx::components
} // hpx

#endif // HPX_352F80CF_5A0C_4DAE_9DC3_8CEE5F789A75

