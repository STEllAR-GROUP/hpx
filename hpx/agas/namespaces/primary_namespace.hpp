////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_AD465B65_6482_4A9E_889D_3EC1FE2E9486)
#define HPX_AD465B65_6482_4A9E_889D_3EC1FE2E9486

#include <map>

#include <hpx/agas/local_address.hpp>
#include <hpx/agas/basic_namespace.hpp>

namespace hpx { namespace agas
{

namespace tag { // hpx::agas::tag

template <typename Protocal>
struct primary_namespace { };

} // hpx::agas::tag

namespace magic { // hpx::agas::magic

template <typename Protocal>
struct registry_type<tag::primary_namespace<Protocal> >
{
    typedef std::map<naming::gid_type,
        typename local_address<Protocal>::registry_entry_type
    > type;
};

// TODO: implement bind_hook, resolve_hook and unbind_hook

} // hpx::agas::magic
} // hpx::agas

namespace components { namespace agas // hpx::components::agas
{

template <typename Protocal>
struct primary_namespace
  : private basic_namespace<tag::primary_namespace<Protocal> >
{
    // TODO: implement interface
};

} // hpx::components::agas
} // hpx::components
} // hpx

#endif // HPX_AD465B65_6482_4A9E_889D_3EC1FE2E9486

