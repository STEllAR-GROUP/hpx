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
#include <hpx/assert.hpp>
#include <hpx/hashing/jenkins_hash.hpp>
#include <hpx/lcos/base_lco_with_value.hpp>
#include <hpx/modules/async_distributed.hpp>
#include <hpx/modules/collectives.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/runtime/agas/server/symbol_namespace.hpp>
#include <hpx/runtime/agas/symbol_namespace.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/type_support/unused.hpp>

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

using hpx::components::component_agas_symbol_namespace;

using hpx::agas::server::symbol_namespace;

HPX_DEFINE_COMPONENT_NAME(symbol_namespace,
    hpx_symbol_namespace);
HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    symbol_namespace, component_agas_symbol_namespace)

HPX_REGISTER_ACTION_ID(
    symbol_namespace::bind_action,
    symbol_namespace_bind_action,
    hpx::actions::symbol_namespace_bind_action_id)

HPX_REGISTER_ACTION_ID(
    symbol_namespace::resolve_action,
    symbol_namespace_resolve_action,
    hpx::actions::symbol_namespace_resolve_action_id)

HPX_REGISTER_ACTION_ID(
    symbol_namespace::unbind_action,
    symbol_namespace_unbind_action,
    hpx::actions::symbol_namespace_unbind_action_id)

HPX_REGISTER_ACTION_ID(
    symbol_namespace::iterate_action,
    symbol_namespace_iterate_action,
    hpx::actions::symbol_namespace_iterate_action_id)

HPX_REGISTER_ACTION_ID(
    symbol_namespace::on_event_action,
    symbol_namespace_on_event_action,
    hpx::actions::symbol_namespace_on_event_action_id)

HPX_REGISTER_ACTION_ID(
    symbol_namespace::statistics_counter_action,
    symbol_namespace_statistics_counter_action,
    hpx::actions::symbol_namespace_statistics_counter_action_id)

namespace hpx { namespace agas
{
    naming::gid_type symbol_namespace::get_service_instance(
        std::uint32_t service_locality_id)
    {
        naming::gid_type service(HPX_AGAS_SYMBOL_NS_MSB, HPX_AGAS_SYMBOL_NS_LSB);
        return naming::replace_locality_id(service, service_locality_id);
    }

    naming::gid_type symbol_namespace::get_service_instance(naming::gid_type const& dest,
        error_code& ec)
    {
        std::uint32_t service_locality_id = naming::get_locality_id_from_gid(dest);
        if (service_locality_id == naming::invalid_locality_id)
        {
            HPX_THROWS_IF(ec, bad_parameter,
                "symbol_namespace::get_service_instance",
                hpx::util::format(
                    "can't retrieve a valid locality id from global address ({1}): ",
                    dest));
            return naming::gid_type();
        }
        return get_service_instance(service_locality_id);
    }

    bool symbol_namespace::is_service_instance(naming::gid_type const& gid)
    {
        return gid.get_lsb() == HPX_AGAS_SYMBOL_NS_LSB &&
            (gid.get_msb() & ~naming::gid_type::locality_id_mask)
            == HPX_AGAS_SYMBOL_NS_MSB;
    }

    naming::id_type symbol_namespace::symbol_namespace_locality(std::string const& key)
    {
        std::uint32_t hash_value = 0;
        if (key.size() < 2 || key[1] != '0' || key[0] != '/')
        {
            // keys starting with '/0' have to go to node 0
            util::jenkins_hash hash;
            hash_value = hash(key) % get_initial_num_localities();
        }
        return naming::id_type(get_service_instance(hash_value),
            naming::id_type::unmanaged);
    }

    symbol_namespace::symbol_namespace()
      : server_(new server::symbol_namespace())
    {}
    symbol_namespace::~symbol_namespace()
    {}

    naming::address::address_type symbol_namespace::ptr() const
    {
        return reinterpret_cast<naming::address::address_type>(server_.get());
    }

    naming::address symbol_namespace::addr() const
    {
        return naming::address(
            hpx::get_locality(),
            components::component_agas_symbol_namespace,
            this->ptr()
        );
    }

    naming::id_type symbol_namespace::gid() const
    {
        return naming::id_type(
            get_service_instance(hpx::get_locality()),
            naming::id_type::unmanaged);
    }

    hpx::future<bool> symbol_namespace::bind_async(std::string key, naming::gid_type gid)
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        naming::id_type dest = symbol_namespace_locality(key);
        if (naming::get_locality_from_gid(dest.get_gid()) == hpx::get_locality())
        {
            return hpx::make_ready_future(server_->bind(std::move(key), std::move(gid)));
        }
        server::symbol_namespace::bind_action action;
        return hpx::async(action, std::move(dest), std::move(key), std::move(gid));
#else
        HPX_UNUSED(key);
        HPX_UNUSED(gid);
        HPX_ASSERT(false);
        return hpx::make_ready_future(true);
#endif
    }

    bool symbol_namespace::bind(std::string key, naming::gid_type gid)
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        naming::id_type dest = symbol_namespace_locality(key);
        if (naming::get_locality_from_gid(dest.get_gid()) == hpx::get_locality())
        {
            return server_->bind(std::move(key), std::move(gid));
        }
        server::symbol_namespace::bind_action action;
        return action(std::move(dest), std::move(key), std::move(gid));
#else
        HPX_UNUSED(key);
        HPX_UNUSED(gid);
        HPX_ASSERT(false);
        return true;
#endif
    }

    hpx::future<naming::id_type> symbol_namespace::resolve_async(std::string key) const
    {
        naming::id_type dest = symbol_namespace_locality(key);
        if (naming::get_locality_from_gid(dest.get_gid()) == hpx::get_locality())
        {
            naming::gid_type raw_gid = server_->resolve(std::move(key));

            if (naming::detail::has_credits(raw_gid))
                return hpx::make_ready_future(
                    naming::id_type(raw_gid, naming::id_type::managed));

            return hpx::make_ready_future(
                naming::id_type(raw_gid, naming::id_type::unmanaged));
        }
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        server::symbol_namespace::resolve_action action;
        return hpx::async(action, std::move(dest), std::move(key));
#else
        HPX_ASSERT(false);
        return hpx::make_ready_future(naming::id_type{});
#endif
    }

    naming::id_type symbol_namespace::resolve(std::string key) const
    {
        return resolve_async(std::move(key)).get();
    }

    hpx::future<naming::id_type> symbol_namespace::unbind_async(std::string key)
    {
        naming::id_type dest = symbol_namespace_locality(key);
        if (naming::get_locality_from_gid(dest.get_gid()) == hpx::get_locality())
        {
            naming::gid_type raw_gid = server_->unbind(std::move(key));

            if (naming::detail::has_credits(raw_gid))
                return hpx::make_ready_future(
                    naming::id_type(raw_gid, naming::id_type::managed));

            return hpx::make_ready_future(
                naming::id_type(raw_gid, naming::id_type::unmanaged));
        }
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        server::symbol_namespace::unbind_action action;
        return hpx::async(action, std::move(dest), std::move(key));
#else
        HPX_ASSERT(false);
        return hpx::make_ready_future(naming::id_type{});
#endif
    }

    naming::id_type symbol_namespace::unbind(std::string key)
    {
        return unbind_async(std::move(key)).get();
    }

    hpx::future<bool> symbol_namespace::on_event(
        std::string const& name
      , bool call_for_past_events
      , hpx::id_type lco
        )
    {
        naming::id_type dest = symbol_namespace_locality(name);
        if (naming::get_locality_from_gid(dest.get_gid()) == hpx::get_locality())
        {
            return hpx::make_ready_future(
                server_->on_event(name, call_for_past_events, std::move(lco)));
        }
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        server::symbol_namespace::on_event_action action;
        return hpx::async(
            action, std::move(dest), name, call_for_past_events, std::move(lco));
#else
        return hpx::make_ready_future(true);
#endif
    }
}}

#if !defined(HPX_COMPUTE_DEVICE_CODE)
typedef symbol_namespace::iterate_action iterate_action;
#endif
HPX_REGISTER_BROADCAST_ACTION_DECLARATION(iterate_action);
HPX_REGISTER_BROADCAST_ACTION(iterate_action);

namespace hpx { namespace agas
{
    hpx::future<symbol_namespace::iterate_names_return_type>
        symbol_namespace::iterate_async(std::string const& pattern) const
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        using return_type = server::symbol_namespace::iterate_names_return_type;

        auto localities = hpx::find_all_localities();
        std::vector<hpx::id_type> symbol_services;
        symbol_services.reserve(localities.size());

        for (auto const& locality : localities)
        {
            auto gid = get_service_instance(
                naming::get_locality_id_from_id(locality));
            symbol_services.push_back(hpx::id_type(gid, hpx::id_type::unmanaged));
        }

        hpx::future<std::vector<return_type> > f =
            lcos::broadcast<iterate_action>(symbol_services, pattern);

        return f.then(
            [](hpx::future<std::vector<return_type> > && f)
            {
                std::vector<return_type> && data = f.get();
                std::map<std::string, hpx::id_type> result;

                for (auto && d : data)
                {
                    for (auto && e : d)
                    {
                        bool has_credits = naming::detail::has_credits(e.second);
                        result[std::move(e.first)] =
                            hpx::id_type(std::move(e.second),
                                has_credits ?
                                    naming::id_type::managed :
                                    naming::id_type::unmanaged);
                    }
                }

                return result;
            });
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
    void symbol_namespace::register_counter_types()
    {
        server::symbol_namespace::register_counter_types();
        server::symbol_namespace::register_global_counter_types();
    }

    void symbol_namespace::register_server_instance(std::uint32_t locality_id)
    {
        std::string str("locality#" +
            std::to_string(locality_id) + "/");
        server_->register_server_instance(str.c_str(), locality_id);
    }

    void symbol_namespace::unregister_server_instance(error_code& ec)
    {
        server_->unregister_server_instance(ec);
    }
}}
