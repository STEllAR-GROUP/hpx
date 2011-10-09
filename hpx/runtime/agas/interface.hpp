////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_A55506A4_4AC7_4FD0_AB0D_ED0D1368FCC5)
#define HPX_A55506A4_4AC7_4FD0_AB0D_ED0D1368FCC5

#include <hpx/config.hpp>

#include <hpx/exception.hpp>
#include <hpx/runtime/naming/name.hpp>

namespace hpx { namespace agas
{

HPX_EXPORT bool register_name(
    std::string const& name
  , naming::id_type const& gid
  , error_code& ec = throws
    );

inline bool register_name(
    std::string const& name
  , naming::gid_type const& gid
  , error_code& ec = throws
    )
{
    naming::id_type tmp(gid, naming::id_type::unmanaged);
    return register_name(name, tmp, ec); 
}

HPX_EXPORT bool unregister_name(
    std::string const& name
  , error_code& ec = throws
    );

HPX_EXPORT bool query_name(
    std::string const& name
  , naming::id_type& gid
  , error_code& ec = throws
    );

HPX_EXPORT bool query_name(
    std::string const& name
  , naming::gid_type& gid
  , error_code& ec = throws
    );

}}

#endif // HPX_A55506A4_4AC7_4FD0_AB0D_ED0D1368FCC5

