////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>
#include <hpx/apply.hpp>
#include <hpx/async.hpp>
#include <hpx/lcos/base_lco_with_value.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/agas/primary_namespace.hpp>
#include <hpx/runtime/agas/server/primary_namespace.hpp>
#include <hpx/runtime/applier/apply_callback.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/serialization/vector.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/format.hpp>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using hpx::components::component_agas_primary_namespace;

using hpx::agas::server::primary_namespace;

HPX_DEFINE_COMPONENT_NAME(primary_namespace,
    hpx_primary_namespace);
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

HPX_REGISTER_BASE_LCO_WITH_VALUE_ID(
    hpx::naming::address, naming_address,
    hpx::actions::base_lco_with_value_naming_address_get,
    hpx::actions::base_lco_with_value_naming_address_set)
HPX_REGISTER_BASE_LCO_WITH_VALUE_ID(
    gva_tuple_type, gva_tuple,
    hpx::actions::base_lco_with_value_gva_tuple_get,
    hpx::actions::base_lco_with_value_gva_tuple_set)
HPX_REGISTER_BASE_LCO_WITH_VALUE_ID(
    std_pair_address_id_type, std_pair_address_id_type,
    hpx::actions::base_lco_with_value_std_pair_address_id_type_get,
    hpx::actions::base_lco_with_value_std_pair_address_id_type_set)
HPX_REGISTER_BASE_LCO_WITH_VALUE_ID(
    std_pair_gid_type, std_pair_gid_type,
    hpx::actions::base_lco_with_value_std_pair_gid_type_get,
    hpx::actions::base_lco_with_value_std_pair_gid_type_set)
HPX_REGISTER_BASE_LCO_WITH_VALUE_ID(
    std::vector<std::int64_t>, vector_std_int64_type,
    hpx::actions::base_lco_with_value_vector_std_int64_get,
    hpx::actions::base_lco_with_value_vector_std_int64_set)

namespace hpx { namespace agas {

    naming::gid_type primary_namespace::get_service_instance(
        std::uint32_t service_locality_id)
    {
        naming::gid_type service(HPX_AGAS_PRIMARY_NS_MSB, HPX_AGAS_PRIMARY_NS_LSB);
        return naming::replace_locality_id(service, service_locality_id);
    }

    naming::gid_type primary_namespace::get_service_instance(naming::gid_type const& dest,
        error_code& ec)
    {
        std::uint32_t service_locality_id = naming::get_locality_id_from_gid(dest);
        if (service_locality_id == naming::invalid_locality_id)
        {
            HPX_THROWS_IF(ec, bad_parameter,
                "primary_namespace::get_service_instance",
                hpx::util::format(
                    "can't retrieve a valid locality id from global address ({1}): ",
                    dest));
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
            hpx::components::component_agas_primary_namespace,
            this->ptr()
        );
    }

    naming::id_type primary_namespace::gid() const
    {
        return naming::id_type(
            get_service_instance(hpx::get_locality()),
            naming::id_type::unmanaged);
    }

    std::pair<naming::id_type, naming::address>
    primary_namespace::begin_migration(naming::gid_type id)
    {
        HPX_ASSERT(
            naming::get_locality_from_gid(get_service_instance(id)) ==
            hpx::get_locality());

        return server_->begin_migration(id);
    }
    bool primary_namespace::end_migration(naming::gid_type id)
    {
        HPX_ASSERT(
            naming::get_locality_from_gid(get_service_instance(id)) ==
            hpx::get_locality());

        return server_->end_migration(id);
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
        naming::gid_type const& id = p.destination();
        naming::id_type dest = naming::id_type(get_service_instance(id),
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
            return hpx::make_ready_future(server_->colocate(id));
        }
        server::primary_namespace::colocate_action action;
        return hpx::async(action, std::move(dest), id);
    }

    future<naming::address>
    primary_namespace::unbind_gid_async(std::uint64_t count, naming::gid_type id)
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
    primary_namespace::unbind_gid(std::uint64_t count, naming::gid_type id)
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

    future<std::int64_t> primary_namespace::increment_credit(
        std::int64_t credits
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
    primary_namespace::allocate(std::uint64_t count)
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

    void primary_namespace::register_server_instance(std::uint32_t locality_id)
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
