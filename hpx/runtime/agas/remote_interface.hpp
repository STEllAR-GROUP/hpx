////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_6D3DC09F_33F9_4B92_BA2F_2209AD532D73)
#define HPX_6D3DC09F_33F9_4B92_BA2F_2209AD532D73

#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/lcos/async.hpp>

namespace hpx { namespace agas { namespace server
{

///////////////////////////////////////////////////////////////////////////////

/// Forwarder.
HPX_EXPORT inline void garbage_collect_non_blocking_remote()
{
    return garbage_collect_non_blocking();
}

typedef actions::plain_action0<garbage_collect_non_blocking_remote>
    garbage_collect_non_blocking_remote_action;

///////////////////////////////////////////////////////////////////////////////

/// Forwarder.
HPX_EXPORT inline void garbage_collect_sync_remote()
{
    return garbage_collect_sync();
}

typedef actions::plain_action0<garbage_collect_sync_remote>
    garbage_collect_sync_remote_action;

}}}

///////////////////////////////////////////////////////////////////////////////

HPX_REGISTER_PLAIN_ACTION_DECLARATION(
    hpx::agas::server::garbage_collect_non_blocking_remote_action
)

HPX_REGISTER_PLAIN_ACTION_DECLARATION(
    hpx::agas::server::garbage_collect_sync_remote_action
)

namespace hpx { namespace agas
{

///////////////////////////////////////////////////////////////////////////////

/// Flush pending reference count requests on \a prefix. Fire-and-forget
/// semantics.
///
/// \param prefix [in] The target locality. 
inline void garbage_collect_non_blocking(
    naming::id_type const& prefix
    )
{
    typedef server::garbage_collect_non_blocking_remote_action action_type;
    applier::apply<action_type>(prefix);
}

/// Flush pending reference count requests on \a prefix. Fire-and-forget
/// semantics.
///
/// \param prefix [in] The target locality. 
inline void garbage_collect_non_blocking(
    naming::gid_type const& prefix
    )
{
    typedef server::garbage_collect_non_blocking_remote_action action_type;
    naming::id_type const tmp(prefix, naming::id_type::unmanaged);
    applier::apply<action_type>(tmp);
}

///////////////////////////////////////////////////////////////////////////////

/// Flush pending reference count requests on \a prefix. Synchronous.
///
/// \param prefix [in] The target locality. 
inline void garbage_collect_sync(
    naming::id_type const& prefix
    )
{
    typedef server::garbage_collect_sync_remote_action action_type;
    lcos::async<action_type>(prefix).get();
}

/// Flush pending reference count requests on \a prefix. Synchronous.
///
/// \param prefix [in] The target locality. 
inline void garbage_collect_sync(
    naming::gid_type const& prefix
    )
{
    typedef server::garbage_collect_sync_remote_action action_type;
    naming::id_type const tmp(prefix, naming::id_type::unmanaged);
    lcos::async<action_type>(tmp).get();
}

}}

#endif

