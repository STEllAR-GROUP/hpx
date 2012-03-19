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
#include <hpx/runtime/agas/response.hpp>

namespace hpx { namespace agas
{

///////////////////////////////////////////////////////////////////////////////
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

HPX_EXPORT lcos::future<bool, response> register_name_async(
    std::string const& name
  , naming::id_type const& id
    );

///////////////////////////////////////////////////////////////////////////////
HPX_EXPORT bool unregister_name(
    std::string const& name
  , error_code& ec = throws
    );

HPX_EXPORT bool unregister_name(
    std::string const& name
  , naming::id_type& gid
  , error_code& ec = throws
    );

HPX_EXPORT lcos::future<naming::id_type, response> unregister_name_async(
    std::string const& name
    );

///////////////////////////////////////////////////////////////////////////////
HPX_EXPORT bool resolve_name(
    std::string const& name
  , naming::id_type& gid
  , error_code& ec = throws
    );

HPX_EXPORT bool resolve_name(
    std::string const& name
  , naming::gid_type& gid
  , error_code& ec = throws
    );

///////////////////////////////////////////////////////////////////////////////
HPX_EXPORT bool is_local_address(
    naming::gid_type const& gid
  , error_code& ec = throws
    );

HPX_EXPORT bool is_local_address(
    naming::gid_type const& gid
  , naming::address& addr
  , error_code& ec = throws
    );

inline bool is_local_address(
    naming::id_type const& gid
  , error_code& ec = throws
    )
{
    return is_local_address(gid.get_gid(), ec);
}

inline bool is_local_address(
    naming::id_type const& gid
  , naming::address& addr
  , error_code& ec = throws
    )
{
    return is_local_address(gid.get_gid(), addr, ec);
}

///////////////////////////////////////////////////////////////////////////////
HPX_EXPORT bool is_local_address_cached(
    naming::gid_type const& gid
  , error_code& ec = throws
    );

HPX_EXPORT bool is_local_address_cached(
    naming::gid_type const& gid
  , naming::address& addr
  , error_code& ec = throws
    );

inline bool is_local_address_cached(
    naming::id_type const& gid
  , error_code& ec = throws
    )
{
    return is_local_address_cached(gid.get_gid(), ec);
}

inline bool is_local_address_cached(
    naming::id_type const& gid
  , naming::address& addr
  , error_code& ec = throws
    )
{
    return is_local_address_cached(gid.get_gid(), addr, ec);
}

///////////////////////////////////////////////////////////////////////////////
HPX_EXPORT bool is_local_lva_encoded_address(
    naming::gid_type const& gid
    );

inline bool is_local_lva_encoded_address(
    naming::id_type const& gid
    )
{
    return is_local_lva_encoded_address(gid.get_gid());
}

///////////////////////////////////////////////////////////////////////////////
HPX_EXPORT void garbage_collect_non_blocking(
    error_code& ec = throws
    );

HPX_EXPORT void garbage_collect(
    error_code& ec = throws
    );

///////////////////////////////////////////////////////////////////////////////
/// \brief Invoke an asynchronous garbage collection step on the given target
///        locality.
HPX_EXPORT void garbage_collect_non_blocking(
    naming::id_type const& id
  , error_code& ec = throws
    );

/// \brief Invoke a synchronous garbage collection step on the given target
///        locality.
HPX_EXPORT void garbage_collect(
    naming::id_type const& id
  , error_code& ec = throws
    );

///////////////////////////////////////////////////////////////////////////////
/// \brief Return an id_type referring to the console locality.
HPX_EXPORT naming::id_type get_console_locality(
    error_code& ec = throws
    );
}}

#endif // HPX_A55506A4_4AC7_4FD0_AB0D_ED0D1368FCC5

