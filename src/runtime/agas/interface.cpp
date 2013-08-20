////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>

#include <algorithm>

namespace hpx { namespace agas
{
///////////////////////////////////////////////////////////////////////////////
bool register_name(
    std::string const& name
  , naming::gid_type const& gid
  , error_code& ec
    )
{
    naming::resolver_client& agas_ = naming::get_agas_client();
    return agas_.register_name(name, gid);
}

bool register_name(
    std::string const& name
  , naming::id_type const& id
  , error_code& ec
    )
{
    naming::resolver_client& agas_ = naming::get_agas_client();
    return agas_.register_name(name, id);
}

lcos::future<bool> register_name_async(
    std::string const& name
  , naming::id_type const& id
    )
{
    naming::resolver_client& agas_ = naming::get_agas_client();
    return agas_.register_name_async(name, id);
}

///////////////////////////////////////////////////////////////////////////////
bool unregister_name(
    std::string const& name
  , naming::id_type& gid
  , error_code& ec
    )
{
    naming::resolver_client& agas_ = naming::get_agas_client();

    naming::gid_type raw_gid;

    if (agas_.unregister_name(name, raw_gid, ec) && !ec)
    {
        if (naming::detail::get_credit_from_gid(raw_gid) != 0)
            gid = naming::id_type(raw_gid, naming::id_type::managed);
        else
            gid = naming::id_type(raw_gid, naming::id_type::unmanaged);

        return true;
    }

    return false;
}

naming::id_type unregister_name(
    std::string const& name
  , error_code& ec
    )
{
    naming::resolver_client& agas_ = naming::get_agas_client();
    return agas_.unregister_name(name);
}

lcos::future<naming::id_type> unregister_name_async(
    std::string const& name
    )
{
    naming::resolver_client& agas_ = naming::get_agas_client();
    return agas_.unregister_name_async(name);
}

///////////////////////////////////////////////////////////////////////////////
bool resolve_name(
    std::string const& name
  , naming::gid_type& gid
  , error_code& ec
    )
{
    naming::resolver_client& agas_ = naming::get_agas_client();

    if (agas_.resolve_name(name, gid, ec) && !ec)
        return true;

    return false;
}

///////////////////////////////////////////////////////////////////////////////
bool resolve_name(
    std::string const& name
  , naming::id_type& gid
  , error_code& ec
    )
{
    naming::resolver_client& agas_ = naming::get_agas_client();

    naming::gid_type raw_gid;

    if (agas_.resolve_name(name, raw_gid, ec) && !ec)
    {
        if (naming::detail::get_credit_from_gid(raw_gid) != 0)
            gid = naming::id_type(raw_gid, naming::id_type::managed);
        else
            gid = naming::id_type(raw_gid, naming::id_type::unmanaged);

        return true;
    }

    return false;
}

lcos::future<naming::id_type> resolve_name_async(
    std::string const& name
    )
{
    naming::resolver_client& agas_ = naming::get_agas_client();
    return agas_.resolve_name_async(name);
}

naming::id_type resolve_name(
    std::string const& name
  , error_code& ec
    )
{
    naming::resolver_client& agas_ = naming::get_agas_client();
    return agas_.resolve_name(name, ec);
}

///////////////////////////////////////////////////////////////////////////////
// lcos::future<std::vector<naming::id_type> > get_localities_async(
//     components::component_type type
//     )
// {
//     naming::resolver_client& agas_ = naming::get_agas_client();
//     return agas_.get_localities_async();
// }
// 
// std::vector<naming::id_type> get_localities(
//     components::component_type type
//   , error_code& ec
//     )
// {
//     naming::resolver_client& agas_ = naming::get_agas_client();
//     return agas_.get_localities(type, ec);
// }

lcos::future<boost::uint32_t> get_num_localities_async(
    components::component_type type
    )
{
    naming::resolver_client& agas_ = naming::get_agas_client();
    return agas_.get_num_localities_async();
}

boost::uint32_t get_num_localities(
    components::component_type type
  , error_code& ec
    )
{
    naming::resolver_client& agas_ = naming::get_agas_client();
    return agas_.get_num_localities(type, ec);
}

lcos::future<std::vector<boost::uint32_t> > get_num_threads_async()
{
    naming::resolver_client& agas_ = naming::get_agas_client();
    return agas_.get_num_threads_async();
}

std::vector<boost::uint32_t> get_num_threads(
    error_code& ec
    )
{
    naming::resolver_client& agas_ = naming::get_agas_client();
    return agas_.get_num_threads(ec);
}

lcos::future<boost::uint32_t> get_num_overall_threads_async()
{
    naming::resolver_client& agas_ = naming::get_agas_client();
    return agas_.get_num_overall_threads_async();
}

boost::uint32_t get_num_overall_threads(
    error_code& ec
    )
{
    naming::resolver_client& agas_ = naming::get_agas_client();
    return agas_.get_num_overall_threads(ec);
}

///////////////////////////////////////////////////////////////////////////////
bool is_local_address(
    naming::gid_type const& gid
  , error_code& ec
    )
{
    return naming::get_agas_client().is_local_address(gid, ec);
}

bool is_local_address(
    naming::gid_type const& gid
  , naming::address& addr
  , error_code& ec
    )
{
    return naming::get_agas_client().is_local_address(gid, addr, ec);
}

inline naming::gid_type const& convert_to_gid(naming::id_type const& id)
{
    return id.get_gid();
}

bool is_local_address(
    std::vector<naming::id_type> const& ids
  , std::vector<naming::address>& addrs
  , boost::dynamic_bitset<>& locals
  , error_code& ec
    )
{
    std::size_t count = ids.size();

    std::vector<naming::gid_type> gids;
    gids.reserve(count);

    std::transform(ids.begin(), ids.end(), std::back_inserter(gids), convert_to_gid);

    addrs.resize(count);
    return naming::get_agas_client().is_local_address(gids.data(), addrs.data(), count, locals, ec);
}

bool is_local_address_cached(
    naming::gid_type const& gid
  , error_code& ec
    )
{
    return naming::get_agas_client().is_local_address_cached(gid, ec);
}

bool is_local_address_cached(
    naming::gid_type const& gid
  , naming::address& addr
  , error_code& ec
    )
{
    return naming::get_agas_client().is_local_address_cached(gid, addr, ec);
}

bool is_local_lva_encoded_address(
    naming::gid_type const& gid
    )
{
    return naming::get_agas_client().is_local_lva_encoded_address(gid.get_msb());
}

void garbage_collect_non_blocking(
    error_code& ec
    )
{
    naming::get_agas_client().garbage_collect_non_blocking(ec);
}

void garbage_collect(
    error_code& ec
    )
{
    naming::get_agas_client().garbage_collect(ec);
}

/// \brief Invoke an asynchronous garbage collection step on the given target
///        locality.
void garbage_collect_non_blocking(
    naming::id_type const& id
  , error_code& ec
    )
{
    try {
        components::stubs::runtime_support::garbage_collect_non_blocking(id);
    }
    catch (hpx::exception const& e) {
        if (&ec == &throws)
            throw;
        ec = make_error_code(e.get_error(), e.what());
    }
}

/// \brief Invoke a synchronous garbage collection step on the given target
///        locality.
void garbage_collect(
    naming::id_type const& id
  , error_code& ec
    )
{
    try {
        components::stubs::runtime_support::garbage_collect(id);
    }
    catch (hpx::exception const& e) {
        if (&ec == &throws)
            throw;
        ec = make_error_code(e.get_error(), e.what());
    }
}

/// \brief Return an id_type referring to the console locality.
naming::id_type get_console_locality(
    error_code& ec
    )
{
    naming::gid_type console;
    naming::get_agas_client().get_console_locality(console, ec);
    if (ec) return naming::invalid_id;

    return naming::id_type(console, naming::id_type::unmanaged);
}

boost::uint32_t get_locality_id(error_code& ec)
{
    if (get_runtime_ptr() == 0)
        return naming::invalid_locality_id;

    naming::gid_type l = naming::get_agas_client().get_local_locality(ec);
    return l ? naming::get_locality_id_from_gid(l) : naming::invalid_locality_id;
}

///////////////////////////////////////////////////////////////////////////////
naming::gid_type get_next_id(
    std::size_t count
  , error_code& ec
    )
{
    if (get_runtime_ptr() == 0)
    {
        HPX_THROWS_IF(ec, invalid_status,
            "get_next_id", "the runtime system has not been started yet.");
        return naming::invalid_gid;
    }

    naming::resolver_client& agas_ = naming::get_agas_client();
    naming::gid_type lower_bound, upper_bound;
    agas_.get_id_range(agas_.get_here(), count, lower_bound, upper_bound, ec);
    if (ec) return naming::invalid_gid;

    return lower_bound;
}
}}

