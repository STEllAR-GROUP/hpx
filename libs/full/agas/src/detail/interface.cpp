//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/agas/addressing_service.hpp>
#include <hpx/assert.hpp>
#include <hpx/components_base/agas_interface.hpp>
#include <hpx/components_base/detail/agas_interface_functions.hpp>
#include <hpx/components_base/pinned_ptr.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime_local/runtime_local.hpp>
#include <hpx/util/generate_unique_ids.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace hpx { namespace agas { namespace detail { namespace impl {

    ///////////////////////////////////////////////////////////////////////////
    bool is_console()
    {
        naming::resolver_client& agas_ = naming::get_agas_client();
        return agas_.is_console();
    }

    ///////////////////////////////////////////////////////////////////////////
    bool register_name(
        std::string const& name, naming::gid_type const& gid, error_code&)
    {
        naming::resolver_client& agas_ = naming::get_agas_client();
        return agas_.register_name(name, gid);
    }

    future<bool> register_name_async(
        std::string const& name, naming::id_type const& id)
    {
        naming::resolver_client& agas_ = naming::get_agas_client();
        return agas_.register_name_async(name, id);
    }

    bool register_name_id(
        std::string const& name, naming::id_type const& id, error_code& ec)
    {
        if (&ec == &throws)
        {
            naming::resolver_client& agas_ = naming::get_agas_client();
            return agas_.register_name(name, id);
        }
        return register_name_async(name, id).get(ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    naming::id_type unregister_name(std::string const& name, error_code&)
    {
        if (!hpx::is_stopped())
        {
            naming::resolver_client& agas_ = naming::get_agas_client();
            return agas_.unregister_name(name);
        }
        return naming::invalid_id;
    }

    future<naming::id_type> unregister_name_async(std::string const& name)
    {
        naming::resolver_client& agas_ = naming::get_agas_client();
        return agas_.unregister_name_async(name);
    }

    ///////////////////////////////////////////////////////////////////////////
    future<naming::id_type> resolve_name_async(std::string const& name)
    {
        naming::resolver_client& agas_ = naming::get_agas_client();
        return agas_.resolve_name_async(name);
    }

    naming::id_type resolve_name(std::string const& name, error_code& ec)
    {
        naming::resolver_client& agas_ = naming::get_agas_client();
        return agas_.resolve_name(name, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    future<std::uint32_t> get_num_localities_async(components::component_type)
    {
        naming::resolver_client& agas_ = naming::get_agas_client();
        return agas_.get_num_localities_async();
    }

    std::uint32_t get_num_localities(
        components::component_type type, error_code& ec)
    {
        naming::resolver_client& agas_ = naming::get_agas_client();
        return agas_.get_num_localities(type, ec);
    }

    future<std::vector<std::uint32_t>> get_num_threads_async()
    {
        naming::resolver_client& agas_ = naming::get_agas_client();
        return agas_.get_num_threads_async();
    }

    std::vector<std::uint32_t> get_num_threads(error_code& ec)
    {
        naming::resolver_client& agas_ = naming::get_agas_client();
        return agas_.get_num_threads(ec);
    }

    future<std::uint32_t> get_num_overall_threads_async()
    {
        naming::resolver_client& agas_ = naming::get_agas_client();
        return agas_.get_num_overall_threads_async();
    }

    std::uint32_t get_num_overall_threads(error_code& ec)
    {
        naming::resolver_client& agas_ = naming::get_agas_client();
        return agas_.get_num_overall_threads(ec);
    }

    std::string get_component_type_name(
        components::component_type type, error_code& ec)
    {
        naming::resolver_client& agas_ = naming::get_agas_client();
        return agas_.get_component_type_name(type, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    bool is_local_address_cached(naming::gid_type const& gid, error_code& ec)
    {
        return naming::get_agas_client().is_local_address_cached(gid, ec);
    }

    bool is_local_address_cached_addr(
        naming::gid_type const& gid, naming::address& addr, error_code& ec)
    {
        return naming::get_agas_client().is_local_address_cached(gid, addr, ec);
    }

    bool is_local_lva_encoded_address(naming::gid_type const& gid)
    {
        return naming::get_agas_client().is_local_lva_encoded_address(
            gid.get_msb());
    }

    ///////////////////////////////////////////////////////////////////////////
    hpx::future<naming::address> resolve_async(naming::id_type const& id)
    {
        naming::resolver_client& agas_ = naming::get_agas_client();
        return agas_.resolve_async(id);
    }

    naming::address resolve(naming::id_type const& id, error_code& ec)
    {
        naming::resolver_client& agas_ = naming::get_agas_client();
        return agas_.resolve_async(id).get(ec);
    }

    hpx::future<bool> bind_async(naming::gid_type const& gid,
        naming::address const& addr, std::uint32_t locality_id)
    {
        naming::resolver_client& agas_ = naming::get_agas_client();
        return agas_.bind_async(gid, addr, locality_id);
    }

    bool bind(naming::gid_type const& gid, naming::address const& addr,
        std::uint32_t locality_id, error_code& ec)
    {
        naming::resolver_client& agas_ = naming::get_agas_client();
        return agas_.bind_async(gid, addr, locality_id).get(ec);
    }

    hpx::future<bool> bind_async_locality(naming::gid_type const& gid,
        naming::address const& addr, naming::gid_type const& locality_)
    {
        naming::resolver_client& agas_ = naming::get_agas_client();
        return agas_.bind_async(gid, addr, locality_);
    }

    bool bind_locality(naming::gid_type const& gid, naming::address const& addr,
        naming::gid_type const& locality_, error_code& ec)
    {
        naming::resolver_client& agas_ = naming::get_agas_client();
        return agas_.bind_async(gid, addr, locality_).get(ec);
    }

    hpx::future<naming::address> unbind_async(
        naming::gid_type const& id, std::uint64_t)
    {
        naming::resolver_client& agas_ = naming::get_agas_client();
        return agas_.unbind_range_async(id);
    }

    naming::address unbind(
        naming::gid_type const& id, std::uint64_t, error_code& ec)
    {
        naming::resolver_client& agas_ = naming::get_agas_client();
        return agas_.unbind_range_async(id).get(ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    void garbage_collect_non_blocking(error_code& ec)
    {
        naming::get_agas_client().garbage_collect_non_blocking(ec);
    }

    void garbage_collect(error_code& ec)
    {
        naming::get_agas_client().garbage_collect(ec);
    }

    /// \brief Invoke an asynchronous garbage collection step on the given target
    ///        locality.
    void garbage_collect_non_blocking_id(
        naming::id_type const& id, error_code& ec)
    {
        try
        {
            components::stubs::runtime_support::garbage_collect_non_blocking(
                id);
        }
        catch (hpx::exception const& e)
        {
            if (&ec == &throws)
                throw;
            ec = make_error_code(e.get_error(), e.what());
        }
    }

    /// \brief Invoke a synchronous garbage collection step on the given target
    ///        locality.
    void garbage_collect_id(naming::id_type const& id, error_code& ec)
    {
        try
        {
            components::stubs::runtime_support::garbage_collect(id);
        }
        catch (hpx::exception const& e)
        {
            if (&ec == &throws)
                throw;
            ec = make_error_code(e.get_error(), e.what());
        }
    }

    /// \brief Return an id_type referring to the console locality.
    naming::id_type get_console_locality(error_code& ec)
    {
        runtime* rt = get_runtime_ptr();
        if (rt == nullptr || rt->get_state() == state_invalid)
        {
            return naming::invalid_id;
        }

        naming::gid_type console;
        naming::get_agas_client().get_console_locality(console, ec);
        if (ec)
        {
            return naming::invalid_id;
        }

        return naming::id_type(console, naming::id_type::unmanaged);
    }

    std::uint32_t get_locality_id(error_code& ec)
    {
        runtime* rt = get_runtime_ptr();
        if (rt == nullptr || rt->get_state() == state_invalid)
        {
            return naming::invalid_locality_id;
        }

        naming::gid_type l = naming::get_agas_client().get_local_locality(ec);
        return l ? naming::get_locality_id_from_gid(l) :
                   naming::invalid_locality_id;
    }

    ///////////////////////////////////////////////////////////////////////////
    naming::gid_type get_next_id(std::size_t count, error_code& ec)
    {
        runtime* rt = get_runtime_ptr();
        if (rt == nullptr)
        {
            HPX_THROWS_IF(ec, invalid_status, "get_next_id",
                "the runtime system has not been started yet.");
            return naming::invalid_gid;
        }

        // during bootstrap we use the id pool
        if (rt->get_state() == state_invalid)
        {
            return hpx::detail::get_next_id(count);
        }

        naming::resolver_client& agas_ = naming::get_agas_client();
        naming::gid_type lower_bound, upper_bound;
        agas_.get_id_range(count, lower_bound, upper_bound, ec);
        if (ec)
        {
            return naming::invalid_gid;
        }
        return lower_bound;
    }

    ///////////////////////////////////////////////////////////////////////////
    void decref(
        naming::gid_type const& gid, std::int64_t credits, error_code& ec)
    {
        naming::resolver_client& resolver = naming::get_agas_client();
        resolver.decref(gid, credits, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    hpx::future<std::int64_t> incref_async(naming::gid_type const& gid,
        std::int64_t credits, naming::id_type const& keep_alive_)
    {
        HPX_ASSERT(!naming::detail::is_locked(gid));

        naming::resolver_client& resolver = naming::get_agas_client();

        if (keep_alive_)
            return resolver.incref_async(gid, credits, keep_alive_);

        naming::id_type keep_alive = naming::id_type(gid, id_type::unmanaged);
        return resolver.incref_async(gid, credits, keep_alive);
    }

    std::int64_t incref(naming::gid_type const& gid, std::int64_t credits,
        naming::id_type const& keep_alive_, error_code&)
    {
        HPX_ASSERT(!naming::detail::is_locked(gid));

        naming::resolver_client& resolver = naming::get_agas_client();

        if (keep_alive_)
        {
            return resolver.incref_async(gid, credits, keep_alive_).get();
        }
        naming::id_type keep_alive = naming::id_type(gid, id_type::unmanaged);
        return resolver.incref_async(gid, credits, keep_alive).get();
    }

    ///////////////////////////////////////////////////////////////////////////
    hpx::future<naming::id_type> get_colocation_id_async(
        naming::id_type const& id)
    {
        naming::resolver_client& resolver = naming::get_agas_client();
        return resolver.get_colocation_id_async(id);
    }

    naming::id_type get_colocation_id(naming::id_type const& id, error_code& ec)
    {
        return get_colocation_id_async(id).get(ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    hpx::future<hpx::id_type> on_symbol_namespace_event(
        std::string const& name, bool call_for_past_events)
    {
        naming::resolver_client& resolver = naming::get_agas_client();
        return resolver.on_symbol_namespace_event(name, call_for_past_events);
    }

    ///////////////////////////////////////////////////////////////////////////
    hpx::future<std::pair<naming::id_type, naming::address>> begin_migration(
        naming::id_type const& id)
    {
        naming::resolver_client& resolver = naming::get_agas_client();
        return resolver.begin_migration(id);
    }

    bool end_migration(naming::id_type const& id)
    {
        naming::resolver_client& resolver = naming::get_agas_client();
        return resolver.end_migration(id);
    }

    hpx::future<void> mark_as_migrated(naming::gid_type const& gid,
        util::unique_function_nonser<std::pair<bool, hpx::future<void>>()>&& f,
        bool expect_to_be_marked_as_migrating)
    {
        naming::resolver_client& resolver = naming::get_agas_client();
        return resolver.mark_as_migrated(
            gid, std::move(f), expect_to_be_marked_as_migrating);
    }

    std::pair<bool, components::pinned_ptr> was_object_migrated(
        naming::gid_type const& gid,
        util::unique_function_nonser<components::pinned_ptr()>&& f)
    {
        naming::resolver_client& resolver = naming::get_agas_client();
        return resolver.was_object_migrated(gid, std::move(f));
    }

    void unmark_as_migrated(naming::gid_type const& gid)
    {
        naming::resolver_client& resolver = naming::get_agas_client();
        return resolver.unmark_as_migrated(gid);
    }

    hpx::future<symbol_namespace::iterate_names_return_type> find_symbols_async(
        std::string const& pattern)
    {
        naming::resolver_client& resolver = naming::get_agas_client();
        return resolver.iterate_ids(pattern);
    }

    symbol_namespace::iterate_names_return_type find_symbols(
        std::string const& pattern)
    {
        return find_symbols_async(pattern).get();
    }

}}}}    // namespace hpx::agas::detail::impl

namespace hpx { namespace agas {

    // initialize AGAS interface function pointers in agas_base module
    struct HPX_EXPORT init_interface_functions
    {
        init_interface_functions()
        {
            detail::is_console = &detail::impl::is_console;

            detail::register_name = &detail::impl::register_name;
            detail::register_name_async = &detail::impl::register_name_async;
            detail::register_name_id = &detail::impl::register_name_id;

            detail::unregister_name_async =
                &detail::impl::unregister_name_async;
            detail::unregister_name = &detail::impl::unregister_name;

            detail::resolve_name_async = &detail::impl::resolve_name_async;
            detail::resolve_name = &detail::impl::resolve_name;

            detail::get_num_localities_async =
                &detail::impl::get_num_localities_async;
            detail::get_num_localities = &detail::impl::get_num_localities;

            detail::get_num_threads_async =
                &detail::impl::get_num_threads_async;
            detail::get_num_threads = &detail::impl::get_num_threads;

            detail::get_num_overall_threads_async =
                &detail::impl::get_num_overall_threads_async;
            detail::get_num_overall_threads =
                &detail::impl::get_num_overall_threads;

            detail::get_component_type_name =
                &detail::impl::get_component_type_name;

            detail::is_local_address_cached =
                &detail::impl::is_local_address_cached;
            detail::is_local_address_cached_addr =
                &detail::impl::is_local_address_cached_addr;
            detail::is_local_lva_encoded_address =
                &detail::impl::is_local_lva_encoded_address;

            detail::resolve_async = &detail::impl::resolve_async;
            detail::resolve = &detail::impl::resolve;

            detail::bind_async = &detail::impl::bind_async;
            detail::bind = &detail::impl::bind;
            detail::bind_async_locality = &detail::impl::bind_async_locality;
            detail::bind_locality = &detail::impl::bind_locality;

            detail::unbind_async = &detail::impl::unbind_async;
            detail::unbind = &detail::impl::unbind;

            detail::garbage_collect_non_blocking =
                &detail::impl::garbage_collect_non_blocking;
            detail::garbage_collect = &detail::impl::garbage_collect;
            detail::garbage_collect_non_blocking_id =
                &detail::impl::garbage_collect_non_blocking_id;
            detail::garbage_collect_id = &detail::impl::garbage_collect_id;

            detail::get_console_locality = &detail::impl::get_console_locality;
            detail::get_locality_id = &detail::impl::get_locality_id;
            detail::get_next_id = &detail::impl::get_next_id;

            detail::decref = &detail::impl::decref;
            detail::incref_async = &detail::impl::incref_async;
            detail::incref = &detail::impl::incref;

            detail::get_colocation_id_async =
                &detail::impl::get_colocation_id_async;
            detail::get_colocation_id = &detail::impl::get_colocation_id;

            detail::on_symbol_namespace_event =
                &detail::impl::on_symbol_namespace_event;

            detail::begin_migration = &detail::impl::begin_migration;
            detail::end_migration = &detail::impl::end_migration;
            detail::mark_as_migrated = &detail::impl::mark_as_migrated;
            detail::was_object_migrated = &detail::impl::was_object_migrated;
            detail::unmark_as_migrated = &detail::impl::unmark_as_migrated;

            detail::find_symbols_async = &detail::impl::find_symbols_async;
            detail::find_symbols = &detail::impl::find_symbols;
        }
    };

    init_interface_functions init;
}}    // namespace hpx::agas
