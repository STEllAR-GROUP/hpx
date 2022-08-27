////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>
#include <hpx/actions_base/component_action.hpp>
#include <hpx/agas_base/server/symbol_namespace.hpp>
#include <hpx/agas_base/symbol_namespace.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_distributed/base_lco_with_value.hpp>
#include <hpx/components_base/agas_interface.hpp>
#include <hpx/hashing/jenkins_hash.hpp>
#include <hpx/modules/async_distributed.hpp>
#include <hpx/type_support/unused.hpp>

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

using hpx::components::component_agas_symbol_namespace;

using hpx::agas::server::symbol_namespace;

HPX_DEFINE_COMPONENT_NAME(symbol_namespace, hpx_symbol_namespace)
HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    symbol_namespace, component_agas_symbol_namespace)

HPX_REGISTER_ACTION_ID(symbol_namespace::bind_action,
    symbol_namespace_bind_action, hpx::actions::symbol_namespace_bind_action_id)

HPX_REGISTER_ACTION_ID(symbol_namespace::resolve_action,
    symbol_namespace_resolve_action,
    hpx::actions::symbol_namespace_resolve_action_id)

HPX_REGISTER_ACTION_ID(symbol_namespace::unbind_action,
    symbol_namespace_unbind_action,
    hpx::actions::symbol_namespace_unbind_action_id)

HPX_REGISTER_ACTION_ID(symbol_namespace::iterate_action,
    symbol_namespace_iterate_action,
    hpx::actions::symbol_namespace_iterate_action_id)

HPX_REGISTER_ACTION_ID(symbol_namespace::on_event_action,
    symbol_namespace_on_event_action,
    hpx::actions::symbol_namespace_on_event_action_id)

namespace hpx { namespace agas {

    naming::gid_type symbol_namespace::get_service_instance(
        std::uint32_t service_locality_id)
    {
        naming::gid_type service(agas::symbol_ns_msb, agas::symbol_ns_lsb);
        return naming::replace_locality_id(service, service_locality_id);
    }

    naming::gid_type symbol_namespace::get_service_instance(
        naming::gid_type const& dest, error_code& ec)
    {
        std::uint32_t service_locality_id =
            naming::get_locality_id_from_gid(dest);
        if (service_locality_id == naming::invalid_locality_id)
        {
            HPX_THROWS_IF(ec, bad_parameter,
                "symbol_namespace::get_service_instance",
                "can't retrieve a valid locality id from global address "
                "({1}): ",
                dest);
            return naming::gid_type();
        }
        return get_service_instance(service_locality_id);
    }

    bool symbol_namespace::is_service_instance(naming::gid_type const& gid)
    {
        return gid.get_lsb() == agas::symbol_ns_lsb &&
            (gid.get_msb() & ~naming::gid_type::locality_id_mask) ==
            agas::symbol_ns_msb;
    }

    hpx::id_type symbol_namespace::symbol_namespace_locality(
        std::string const& key)
    {
        std::uint32_t hash_value = 0;
        if (key.size() < 2 || key[1] != '0' || key[0] != '/')
        {
            // keys starting with '/0' have to go to node 0
            util::jenkins_hash hash;
            hash_value = hash(key) % get_initial_num_localities();
        }
        return hpx::id_type(get_service_instance(hash_value),
            hpx::id_type::management_type::unmanaged);
    }

    symbol_namespace::symbol_namespace()
      : server_(new server::symbol_namespace())
    {
    }

    naming::address_type symbol_namespace::ptr() const
    {
        return reinterpret_cast<naming::address::address_type>(server_.get());
    }

    naming::address symbol_namespace::addr() const
    {
        return naming::address(agas::get_locality(),
            components::component_agas_symbol_namespace, this->ptr());
    }

    hpx::id_type symbol_namespace::gid() const
    {
        return hpx::id_type(get_service_instance(agas::get_locality()),
            hpx::id_type::management_type::unmanaged);
    }

    hpx::future<bool> symbol_namespace::bind_async(
        std::string key, naming::gid_type gid)
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        hpx::id_type dest = symbol_namespace_locality(key);

        if (naming::get_locality_id_from_id(dest) == agas::get_locality_id())
        {
            return hpx::make_ready_future(
                server_->bind(HPX_MOVE(key), HPX_MOVE(gid)));
        }
        server::symbol_namespace::bind_action action;
        return hpx::async(action, HPX_MOVE(dest), HPX_MOVE(key), HPX_MOVE(gid));
#else
        HPX_UNUSED(key);
        HPX_UNUSED(gid);
        HPX_ASSERT(false);
        return hpx::make_ready_future(true);
#endif
    }

    bool symbol_namespace::bind(std::string key, naming::gid_type gid)
    {
        hpx::id_type dest = symbol_namespace_locality(key);

        if (naming::get_locality_id_from_id(dest) == agas::get_locality_id())
        {
            return server_->bind(HPX_MOVE(key), HPX_MOVE(gid));
        }

#if !defined(HPX_COMPUTE_DEVICE_CODE)
        server::symbol_namespace::bind_action action;
        return action(HPX_MOVE(dest), HPX_MOVE(key), HPX_MOVE(gid));
#else
        HPX_UNUSED(key);
        HPX_UNUSED(gid);
        HPX_ASSERT(false);
        return true;
#endif
    }

    hpx::future<hpx::id_type> symbol_namespace::resolve_async(
        std::string key) const
    {
        hpx::id_type dest = symbol_namespace_locality(key);

        if (naming::get_locality_id_from_id(dest) == agas::get_locality_id())
        {
            naming::gid_type raw_gid = server_->resolve(HPX_MOVE(key));

            if (naming::detail::has_credits(raw_gid))
                return hpx::make_ready_future(hpx::id_type(
                    raw_gid, hpx::id_type::management_type::managed));

            return hpx::make_ready_future(hpx::id_type(
                raw_gid, hpx::id_type::management_type::unmanaged));
        }
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        server::symbol_namespace::resolve_action action;
        return hpx::async(action, HPX_MOVE(dest), HPX_MOVE(key));
#else
        HPX_ASSERT(false);
        return hpx::make_ready_future(hpx::id_type{});
#endif
    }

    hpx::id_type symbol_namespace::resolve(std::string key) const
    {
        return resolve_async(HPX_MOVE(key)).get();
    }

    hpx::future<hpx::id_type> symbol_namespace::unbind_async(std::string key)
    {
        hpx::id_type dest = symbol_namespace_locality(key);

        if (naming::get_locality_id_from_id(dest) == agas::get_locality_id())
        {
            naming::gid_type raw_gid = server_->unbind(HPX_MOVE(key));

            if (naming::detail::has_credits(raw_gid))
                return hpx::make_ready_future(hpx::id_type(
                    raw_gid, hpx::id_type::management_type::managed));

            return hpx::make_ready_future(hpx::id_type(
                raw_gid, hpx::id_type::management_type::unmanaged));
        }
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        server::symbol_namespace::unbind_action action;
        return hpx::async(action, HPX_MOVE(dest), HPX_MOVE(key));
#else
        HPX_ASSERT(false);
        return hpx::make_ready_future(hpx::id_type{});
#endif
    }

    hpx::id_type symbol_namespace::unbind(std::string key)
    {
        return unbind_async(HPX_MOVE(key)).get();
    }

    hpx::future<bool> symbol_namespace::on_event(
        std::string const& name, bool call_for_past_events, hpx::id_type lco)
    {
        hpx::id_type dest = symbol_namespace_locality(name);

        if (naming::get_locality_id_from_id(dest) == agas::get_locality_id())
        {
            return hpx::make_ready_future(
                server_->on_event(name, call_for_past_events, HPX_MOVE(lco)));
        }
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        server::symbol_namespace::on_event_action action;
        return hpx::async(
            action, HPX_MOVE(dest), name, call_for_past_events, HPX_MOVE(lco));
#else
        return hpx::make_ready_future(true);
#endif
    }
}}    // namespace hpx::agas

namespace hpx { namespace agas {

    hpx::future<symbol_namespace::iterate_names_return_type>
    symbol_namespace::iterate_async(std::string const& pattern) const
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        using return_type = server::symbol_namespace::iterate_names_return_type;

        std::vector<std::uint32_t> ids = agas::get_all_locality_ids();

        std::vector<hpx::future<return_type>> results;
        results.reserve(ids.size());
        for (auto id : ids)
        {
            hpx::id_type target(get_service_instance(id),
                hpx::id_type::management_type::unmanaged);

            using iterate_action = server::symbol_namespace::iterate_action;
            results.push_back(
                hpx::async(iterate_action(), HPX_MOVE(target), pattern));
        }

        return hpx::dataflow(
            hpx::unwrapping([](std::vector<return_type>&& data) {
                std::map<std::string, hpx::id_type> result;
                for (auto&& d : data)
                {
                    for (auto&& e : d)
                    {
                        bool has_credits =
                            naming::detail::has_credits(e.second);
                        result[HPX_MOVE(e.first)] =
                            hpx::id_type(HPX_MOVE(e.second),
                                has_credits ?
                                    hpx::id_type::management_type::managed :
                                    hpx::id_type::management_type::unmanaged);
                    }
                }
                return result;
            }),
            results);
#else
        HPX_UNUSED(pattern);
        HPX_ASSERT(false);
        return hpx::make_ready_future(
            symbol_namespace::iterate_names_return_type{});
#endif
    }

    symbol_namespace::iterate_names_return_type symbol_namespace::iterate(
        std::string const& pattern) const
    {
        return iterate_async(pattern).get();
    }

    ///////////////////////////////////////////////////////////////////////////
    void symbol_namespace::register_server_instance(std::uint32_t locality_id)
    {
        std::string str("locality#" + std::to_string(locality_id) + "/");
        server_->register_server_instance(str.c_str(), locality_id);
    }

    void symbol_namespace::unregister_server_instance(error_code& ec)
    {
        server_->unregister_server_instance(ec);
    }
}}    // namespace hpx::agas
