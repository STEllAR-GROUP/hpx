////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>
#include <hpx/apply.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/lcos/base_lco_with_value.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/agas/primary_namespace.hpp>
#include <hpx/runtime/agas/server/primary_namespace.hpp>
#include <hpx/runtime/applier/apply_callback.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/serialization/vector.hpp>

#include <boost/format.hpp>

#include <string>
#include <utility>

using hpx::components::component_agas_primary_namespace;

using hpx::agas::server::primary_namespace;

HPX_REGISTER_COMPONENT(
    hpx::components::fixed_component<primary_namespace>,
    primary_namespace, hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    primary_namespace, component_agas_primary_namespace)

HPX_REGISTER_ACTION_ID(
    primary_namespace::allocate_action,
    primary_namespace_allocate_action,
    hpx::actions::primary_namespace_allocate_action_id)

HPX_REGISTER_ACTION_ID(
    primary_namespace::bind_gid_action,
    primary_namespace_bind_gid_action,
    hpx::actions::primary_namespace_bind_gid_action_id)

HPX_REGISTER_ACTION_ID(
    primary_namespace::begin_migration_action,
    primary_namespace_begin_migration_action,
    hpx::actions::primary_namespace_begin_migration_action_id)

HPX_REGISTER_ACTION_ID(
    primary_namespace::end_migration_action,
    primary_namespace_end_migration_action,
    hpx::actions::primary_namespace_end_migration_action_id)

HPX_REGISTER_ACTION_ID(
    primary_namespace::decrement_credit_action,
    primary_namespace_decrement_credit_action,
    hpx::actions::primary_namespace_decrement_credit_action_id)

HPX_REGISTER_ACTION_ID(
    primary_namespace::increment_credit_action,
    primary_namespace_increment_credit_action,
    hpx::actions::primary_namespace_increment_credit_action_id)

HPX_REGISTER_ACTION_ID(
    primary_namespace::resolve_gid_action,
    primary_namespace_resolve_gid_action,
    hpx::actions::primary_namespace_resolve_gid_action_id)

HPX_REGISTER_ACTION_ID(
    primary_namespace::colocate_action,
    primary_namespace_colocate_action,
    hpx::actions::primary_namespace_colocate_action_id)

HPX_REGISTER_ACTION_ID(
    primary_namespace::unbind_gid_action,
    primary_namespace_unbind_gid_action,
    hpx::actions::primary_namespace_unbind_gid_action_id)

HPX_REGISTER_ACTION_ID(
    primary_namespace::route_action,
    primary_namespace_route_action,
    hpx::actions::primary_namespace_route_action_id)

HPX_REGISTER_ACTION_ID(
    primary_namespace::statistics_counter_action,
    primary_namespace_statistics_counter_action,
    hpx::actions::primary_namespace_statistics_counter_action_id)

namespace hpx { namespace agas {

    naming::gid_type primary_namespace::get_service_instance(
        boost::uint32_t service_locality_id)
    {
        naming::gid_type service(HPX_AGAS_PRIMARY_NS_MSB, HPX_AGAS_PRIMARY_NS_LSB);
        return naming::replace_locality_id(service, service_locality_id);
    }

    naming::gid_type primary_namespace::get_service_instance(naming::gid_type const& dest,
        error_code& ec)
    {
        boost::uint32_t service_locality_id = naming::get_locality_id_from_gid(dest);
        if (service_locality_id == naming::invalid_locality_id)
        {
            HPX_THROWS_IF(ec, bad_parameter,
                "primary_namespace::get_service_instance",
                boost::str(boost::format(
                        "can't retrieve a valid locality id from global address (%1%): "
                    ) % dest));
            return naming::gid_type();
        }
        return get_service_instance(service_locality_id);
    }

    bool primary_namespace::is_service_instance(naming::gid_type const& gid)
    {
        return gid.get_lsb() == HPX_AGAS_PRIMARY_NS_LSB &&
            (gid.get_msb() & ~naming::gid_type::locality_id_mask)
            == HPX_AGAS_PRIMARY_NS_MSB;
    }

    primary_namespace::primary_namespace()
      : server_(new server::primary_namespace())
    {}

    primary_namespace::~primary_namespace()
    {}

    naming::address::address_type primary_namespace::ptr() const
    {
        return reinterpret_cast<naming::address::address_type>(server_.get());
    }

    naming::address primary_namespace::addr() const
    {
        return naming::address(
            hpx::get_locality(),
            server::primary_namespace::get_component_type(),
            this->ptr()
        );
    }

    naming::id_type primary_namespace::gid() const
    {
        return naming::id_type(
            get_service_instance(hpx::get_locality()),
            naming::id_type::unmanaged);
    }

    future<std::pair<naming::id_type, naming::address>>
    primary_namespace::begin_migration(naming::gid_type id)
    {
        naming::id_type dest = naming::id_type(get_service_instance(id),
            naming::id_type::unmanaged);
        if (naming::get_locality_from_gid(dest.get_gid()) == hpx::get_locality())
        {
            return hpx::make_ready_future(server_->begin_migration(id));
        }
        server::primary_namespace::begin_migration_action action;
        return hpx::async(action, std::move(dest), id);
    }
    future<bool> primary_namespace::end_migration(naming::gid_type id)
    {
        naming::id_type dest = naming::id_type(get_service_instance(id),
            naming::id_type::unmanaged);
        if (naming::get_locality_from_gid(dest.get_gid()) == hpx::get_locality())
        {
            return hpx::make_ready_future(server_->end_migration(id));
        }
        server::primary_namespace::end_migration_action action;
        return hpx::async(action, std::move(dest), id);
    }

    bool primary_namespace::bind_gid(
        gva g, naming::gid_type id, naming::gid_type locality)
    {
        return server_->bind_gid(g, id, locality);
    }

    future<bool> primary_namespace::bind_gid_async(
        gva g, naming::gid_type id, naming::gid_type locality)
    {
        naming::id_type dest = naming::id_type(get_service_instance(id),
            naming::id_type::unmanaged);
        if (naming::get_locality_from_gid(dest.get_gid()) == hpx::get_locality())
        {
            return hpx::make_ready_future(server_->bind_gid(g, id, locality));
        }
        server::primary_namespace::bind_gid_action action;
        return hpx::async(action, std::move(dest), g, id, locality);
    }

    void primary_namespace::route(parcelset::parcel && p,
        util::function_nonser<void(boost::system::error_code const&,
        parcelset::parcel const&)> && f)
    {
        // compose request
        naming::id_type const* ids = p.destinations();
        naming::id_type dest = naming::id_type(get_service_instance(ids[0]),
            naming::id_type::unmanaged);
        if (naming::get_locality_from_gid(dest.get_gid()) == hpx::get_locality())
        {
            hpx::apply(
                &server::primary_namespace::route,
                server_.get(),
                std::move(p)
            );
            f(boost::system::error_code(), parcelset::parcel());
            return;
        }

        server::primary_namespace::route_action action;
        hpx::apply_cb(action, std::move(dest), std::move(f), std::move(p));
    }

    primary_namespace::resolved_type
    primary_namespace::resolve_gid(naming::gid_type id)
    {
        return server_->resolve_gid(id);
    }

    future<primary_namespace::resolved_type>
    primary_namespace::resolve_full(naming::gid_type id)
    {
        naming::id_type dest = naming::id_type(get_service_instance(id),
            naming::id_type::unmanaged);
        if (naming::get_locality_from_gid(dest.get_gid()) == hpx::get_locality())
        {
            return hpx::make_ready_future(server_->resolve_gid(id));
        }
        server::primary_namespace::resolve_gid_action action;
        return hpx::async(action, std::move(dest), id);
    }

    hpx::future<id_type> primary_namespace::colocate(naming::gid_type id)
    {
        naming::id_type dest = naming::id_type(get_service_instance(id),
            naming::id_type::unmanaged);
        if (naming::get_locality_from_gid(dest.get_gid()) == hpx::get_locality())
        {
            return hpx::make_ready_future(
                naming::id_type(server_->colocate(id), naming::id_type::unmanaged));
        }
        server::primary_namespace::colocate_action action;
        return hpx::async(action, std::move(dest), id);
    }

    future<naming::address>
    primary_namespace::unbind_gid_async(boost::uint64_t count, naming::gid_type id)
    {
        naming::id_type dest = naming::id_type(get_service_instance(id),
            naming::id_type::unmanaged);
        naming::gid_type stripped_id = naming::detail::get_stripped_gid(id);
        if (naming::get_locality_from_gid(dest.get_gid()) == hpx::get_locality())
        {
            return hpx::make_ready_future(server_->unbind_gid(count, stripped_id));
        }
        server::primary_namespace::unbind_gid_action action;
        return hpx::async(action, std::move(dest), count, stripped_id);
    }

    naming::address
    primary_namespace::unbind_gid(boost::uint64_t count, naming::gid_type id)
    {
        naming::id_type dest = naming::id_type(get_service_instance(id),
            naming::id_type::unmanaged);
        naming::gid_type stripped_id = naming::detail::get_stripped_gid(id);
        if (naming::get_locality_from_gid(dest.get_gid()) == hpx::get_locality())
        {
            return server_->unbind_gid(count, stripped_id);
        }
        server::primary_namespace::unbind_gid_action action;
        return action(std::move(dest), count, stripped_id);
    }

    future<boost::int64_t> primary_namespace::increment_credit(
        boost::int64_t credits
      , naming::gid_type lower
      , naming::gid_type upper
        )
    {
        naming::id_type dest = naming::id_type(get_service_instance(lower),
            naming::id_type::unmanaged);
        if (naming::get_locality_from_gid(dest.get_gid()) == hpx::get_locality())
        {
            return hpx::make_ready_future(
                server_->increment_credit(credits, lower, upper));
        }
        server::primary_namespace::increment_credit_action action;
        return hpx::async(action, std::move(dest), credits, lower, upper);
    }

    std::pair<naming::gid_type, naming::gid_type>
    primary_namespace::allocate(boost::uint64_t count)
    {
        return server_->allocate(count);
    }

    void primary_namespace::set_local_locality(naming::gid_type const& g)
    {
        server_->set_local_locality(g);
    }

    void primary_namespace::register_counter_types()
    {
        server::primary_namespace::register_counter_types();
        server::primary_namespace::register_global_counter_types();
    }

    void primary_namespace::register_server_instance(boost::uint32_t locality_id)
    {
        std::string str("locality#" +
            std::to_string(locality_id) + "/");
        server_->register_server_instance(str.c_str(), locality_id);
    }

    void primary_namespace::unregister_server_instance(error_code& ec)
    {
        server_->unregister_server_instance(ec);
    }
}}
