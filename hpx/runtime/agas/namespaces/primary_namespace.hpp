////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_AD465B65_6482_4A9E_889D_3EC1FE2E9486)
#define HPX_AD465B65_6482_4A9E_889D_3EC1FE2E9486

#include <map>

#include <hpx/runtime/agas/traits.hpp>
#include <hpx/runtime/agas/local_address.hpp>
#include <hpx/runtime/agas/basic_namespace.hpp>

namespace hpx { namespace agas // hpx::agas
{

namespace tag { // hpx::agas::tag

template <typename Protocal>
struct primary_namespace;

} // hpx::agas::tag

namespace traits { // hpx::agas::traits

template <typename Protocal>
struct registry_type<tag::primary_namespace<Protocal> >
{
    typedef std::map<naming::gid_type,
        typename local_address<Protocal>::registry_entry_type
    > type;
};

// TODO: implement bind_hook, update_hook, resolve_hook and unbind_hook

} // hpx::agas::traits
} // hpx::agas

///////////////////////////////////////////////////////////////////////////////
namespace components { namespace agas // hpx::components::agas
{

template <typename Protocal>
struct primary_namespace
  : private basic_namespace<hpx::agas::tag::primary_namespace<Protocal> >
{
    // TODO: implement interface
};

// MPL metafunction (syntactic sugar)
template <typename Protocal>
struct primary_namespace_type
{ typedef primary_namespace<Protocal> type; }; 

///////////////////////////////////////////////////////////////////////////////
namespace server // hpx::components::agas::server
{

// MPL metafunction
template <typename Protocal>
struct primary_namespace_type
{
    typedef server::basic_namespace<
        hpx::agas::tag::primary_namespace<Protocal>
    > type
};

} // hpx::components::agas::server

///////////////////////////////////////////////////////////////////////////////
namespace stubs // hpx::components::agas::stubs
{

// MPL metafunction
template <typename Protocal>
struct primary_namespace_type
{
    typedef stubs::basic_namespace<
        hpx::agas::tag::primary_namespace<Protocal>
    > type
};

} // hpx::components::agas::stubs

} // hpx::components::agas
} // hpx::components
} // hpx

#endif // HPX_AD465B65_6482_4A9E_889D_3EC1FE2E9486

