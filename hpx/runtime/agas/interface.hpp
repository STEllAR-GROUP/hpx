////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>

#include <hpx/async_base/launch_policy.hpp>
#include <hpx/components_base/component_type.hpp>
#include <hpx/functional/unique_function.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/runtime/components_fwd.hpp>

#include <boost/dynamic_bitset.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace hpx { namespace agas
{

///////////////////////////////////////////////////////////////////////////////
HPX_EXPORT bool is_console();

///////////////////////////////////////////////////////////////////////////////
HPX_EXPORT bool register_name(
    launch::sync_policy
  , std::string const& name
  , naming::gid_type const& gid
  , error_code& ec = throws
    );

HPX_EXPORT bool register_name(
    launch::sync_policy
  , std::string const& name
  , naming::id_type const& id
  , error_code& ec = throws
    );

HPX_EXPORT lcos::future<bool> register_name(
    std::string const& name
  , naming::id_type const& id
    );

///////////////////////////////////////////////////////////////////////////////
HPX_EXPORT naming::id_type unregister_name(
    launch::sync_policy
  , std::string const& name
  , error_code& ec = throws
    );

HPX_EXPORT lcos::future<naming::id_type> unregister_name(
    std::string const& name
    );

///////////////////////////////////////////////////////////////////////////////
HPX_EXPORT naming::id_type resolve_name(
    launch::sync_policy
  , std::string const& name
  , error_code& ec = throws
    );

HPX_EXPORT lcos::future<naming::id_type> resolve_name(
    std::string const& name
    );

///////////////////////////////////////////////////////////////////////////////
// HPX_EXPORT lcos::future<std::vector<naming::id_type> > get_localities(
//     components::component_type type = components::component_invalid
//     );
//
// HPX_EXPORT std::vector<naming::id_type> get_localities_sync(
//     components::component_type type
//   , error_code& ec = throws
//     );
//
// inline std::vector<naming::id_type> get_localities_sync(
//     error_code& ec = throws
//     )
// {
//     return get_localities(components::component_invalid, ec);
// }

///////////////////////////////////////////////////////////////////////////////
HPX_EXPORT lcos::future<std::uint32_t> get_num_localities(
    components::component_type type = components::component_invalid
    );

HPX_EXPORT std::uint32_t get_num_localities(
    launch::sync_policy
  , components::component_type type
  , error_code& ec = throws
    );

inline std::uint32_t get_num_localities(
    launch::sync_policy
  , error_code& ec = throws
    )
{
    return hpx::agas::get_num_localities(launch::sync,
        components::component_invalid, ec);
}

///////////////////////////////////////////////////////////////////////////////
HPX_EXPORT std::string get_component_type_name(
    components::component_type type, error_code& ec = throws);

///////////////////////////////////////////////////////////////////////////////
HPX_EXPORT lcos::future<std::vector<std::uint32_t> > get_num_threads();

HPX_EXPORT std::vector<std::uint32_t> get_num_threads(
    launch::sync_policy
  , error_code& ec = throws
    );

///////////////////////////////////////////////////////////////////////////////
HPX_EXPORT lcos::future<std::uint32_t> get_num_overall_threads();

HPX_EXPORT std::uint32_t get_num_overall_threads(
    launch::sync_policy
  , error_code& ec = throws
    );

///////////////////////////////////////////////////////////////////////////////
HPX_EXPORT std::uint32_t get_locality_id(error_code& ec = throws);

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
HPX_EXPORT hpx::future<naming::address> resolve(
    naming::id_type const& id
    );

HPX_EXPORT naming::address resolve(
    launch::sync_policy
  , naming::id_type const& id
  , error_code& ec = throws
    );

HPX_EXPORT hpx::future<bool> bind(
    naming::gid_type const& gid
  , naming::address const& addr
  , std::uint32_t locality_id
    );

HPX_EXPORT bool bind(
    launch::sync_policy
  , naming::gid_type const& gid
  , naming::address const& addr
  , std::uint32_t locality_id
  , error_code& ec = throws
    );

HPX_EXPORT hpx::future<bool> bind(
    naming::gid_type const& gid
  , naming::address const& addr
  , naming::gid_type const& locality_
    );

HPX_EXPORT bool bind(
    launch::sync_policy
  , naming::gid_type const& gid
  , naming::address const& addr
  , naming::gid_type const& locality_
  , error_code& ec = throws
    );

HPX_EXPORT hpx::future<naming::address> unbind(
    naming::gid_type const& gid
  , std::uint64_t count = 1
    );

HPX_EXPORT naming::address unbind(
    launch::sync_policy
  , naming::gid_type const& gid
  , std::uint64_t count = 1
  , error_code& ec = throws
    );

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

///////////////////////////////////////////////////////////////////////////////
HPX_EXPORT naming::gid_type get_next_id(
    std::size_t count
  , error_code& ec = throws
    );

///////////////////////////////////////////////////////////////////////////////
HPX_EXPORT void decref(
    naming::gid_type const& id
  , std::int64_t credits
  , error_code& ec = throws
  );

///////////////////////////////////////////////////////////////////////////////
HPX_EXPORT hpx::future<std::int64_t> incref(
    naming::gid_type const& gid
  , std::int64_t credits
  , naming::id_type const& keep_alive = naming::invalid_id
  );

HPX_EXPORT std::int64_t incref(
    launch::sync_policy
  , naming::gid_type const& gid
  , std::int64_t credits = 1
  , naming::id_type const& keep_alive = naming::invalid_id
  , error_code& ec = throws
    );

///////////////////////////////////////////////////////////////////////////////
HPX_EXPORT hpx::future<naming::id_type> get_colocation_id(
    naming::id_type const& id);

HPX_EXPORT naming::id_type get_colocation_id(
    launch::sync_policy
  , naming::id_type const& id
  , error_code& ec = throws);

///////////////////////////////////////////////////////////////////////////////
HPX_EXPORT hpx::future<hpx::id_type> on_symbol_namespace_event(
    std::string const& name, bool call_for_past_events);

///////////////////////////////////////////////////////////////////////////////
HPX_EXPORT hpx::future<std::pair<naming::id_type, naming::address>>
    begin_migration(naming::id_type const& id);
HPX_EXPORT bool end_migration(naming::id_type const& id);

HPX_EXPORT hpx::future<void>
    mark_as_migrated(naming::gid_type const& gid,
        util::unique_function_nonser<
            std::pair<bool, hpx::future<void> >()> && f,
        bool expect_to_be_marked_as_migrating);

HPX_EXPORT std::pair<bool, components::pinned_ptr>
    was_object_migrated(naming::gid_type const& gid,
        util::unique_function_nonser<components::pinned_ptr()> && f);

HPX_EXPORT void unmark_as_migrated(naming::gid_type const& gid);

HPX_EXPORT hpx::future<std::map<std::string, hpx::id_type> >
    find_symbols(std::string const& pattern = "*");
HPX_EXPORT std::map<std::string, hpx::id_type> find_symbols(
    hpx::launch::sync_policy, std::string const& pattern = "*");
}}


