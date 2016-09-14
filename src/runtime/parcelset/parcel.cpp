//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/runtime_support.hpp>
#include <hpx/runtime/agas/addressing_service.hpp>
#include <hpx/runtime/parcelset/parcel.hpp>
#include <hpx/runtime/parcelset/parcelhandler.hpp>
#include <hpx/runtime/serialization/access.hpp>
#include <hpx/runtime/serialization/input_archive.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>
#include <hpx/runtime/serialization/detail/polymorphic_id_factory.hpp>

#include <boost/atomic.hpp>

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace detail
{
    void dijkstra_make_black();     // forward declaration only
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset
{
    namespace detail
    {
        // The original parcel-sent handler is wrapped to keep the parcel alive
        // until after the data has been reliably sent (which is needed for zero
        // copy serialization).
        void parcel_sent_handler(
            boost::system::error_code const& ec,
            parcelset::parcel const& p)
        {
            parcelhandler& ph = hpx::get_runtime().get_parcel_handler();
            // invoke the original handler
            ph.invoke_write_handler(ec, p);

            // inform termination detection of a sent message
            if (!p.does_termination_detection())
                hpx::detail::dijkstra_make_black();
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // generate unique parcel id
    naming::gid_type parcel::generate_unique_id(
        std::uint32_t locality_id_default)
    {
        static boost::atomic<std::uint64_t> id(0);

        error_code ec(lightweight);        // ignore all errors
        std::uint32_t locality_id = hpx::get_locality_id(ec);
        if (locality_id == naming::invalid_locality_id)
            locality_id = locality_id_default;

        naming::gid_type result = naming::get_gid_from_locality_id(locality_id);
        result.set_lsb(++id);
        return result;
    }

    naming::address_type parcel::determine_lva()
    {
        naming::resolver_client& client = hpx::naming::get_agas_client();
        int comptype = action_->get_component_type();

        // decode the local virtual address of the parcel
        naming::address::address_type lva = data_.addr_.address_;
        // by convention, a zero address references either the local
        // runtime support component or one of the AGAS components
        if (0 == lva)
        {
            switch(comptype)
            {
            case components::component_runtime_support:
                lva = hpx::applier::get_applier().get_runtime_support_raw_gid().get_lsb();
                break;

            case components::component_agas_primary_namespace:
                lva = client.get_primary_ns_lva();
                break;

            case components::component_agas_symbol_namespace:
                lva = client.get_symbol_ns_lva();
                break;

            case components::component_plain_function:
                break;

            default:
                HPX_ASSERT(false);
            }
        }
        else if (comptype == components::component_memory)
        {
            HPX_ASSERT(naming::refers_to_virtual_memory(data_.dest_));
            lva = hpx::applier::get_applier().get_memory_raw_gid().get_lsb();
        }

        // make sure the component_type of the action matches the
        // component type in the destination address
        if (HPX_UNLIKELY(!components::types_are_compatible(
            data_.addr_.type_, comptype)))
        {
            std::ostringstream strm;
            strm << " types are not compatible: destination_type("
                  << data_.addr_.type_ << ") action_type(" << comptype
                  << ") parcel ("  << *this << ")";
            HPX_THROW_EXCEPTION(bad_component_type,
                "applier::schedule_action",
                strm.str());
        }

        return lva;
    }

    void parcel::load_schedule(serialization::input_archive & ar,
        std::size_t num_thread)
    {
        load_data(ar);
        // make sure this parcel destination matches the proper locality
        HPX_ASSERT(destination_locality() == data_.addr_.locality_);

        naming::address_type lva = determine_lva();

        // make sure the target has not been migrated away
        auto r = action_->was_object_migrated(data_.dest_, lva);
        if (r.first)
        {
            naming::resolver_client& client = hpx::naming::get_agas_client();
            // If the object was migrated, just load the action and route.
            ar >> *action_;
            client.route(
                std::move(*this),
                &detail::parcel_sent_handler,
                threads::thread_priority_normal);
            return;
        }

        // dispatch action, register work item either with or without
        // continuation support
        if (cont_)
        {
            // No continuation is to be executed, register the plain
            // action and the local-virtual address.
            action_->load_schedule(ar, std::move(cont_), std::move(data_.dest_),
                lva, num_thread);
        }
        else
        {
            // This parcel carries a continuation, register a wrapper
            // which first executes the original thread function as
            // required by the action and triggers the continuations
            // afterwards.
            action_->load_schedule(ar, std::move(data_.dest_), lva, num_thread);
        }
    }

    void parcel::schedule_action()
    {
        // make sure this parcel destination matches the proper locality
        HPX_ASSERT(destination_locality() == data_.addr_.locality_);

        naming::address_type lva = determine_lva();

        // make sure the target has not been migrated away
        auto r = action_->was_object_migrated(data_.dest_, lva);
        if (r.first)
        {
            naming::resolver_client& client = hpx::naming::get_agas_client();
            // If the object was migrated, just route.
            client.route(
                std::move(*this),
                &detail::parcel_sent_handler,
                threads::thread_priority_normal);
            return;
        }

        // dispatch action, register work item either with or without
        // continuation support
        if (cont_)
        {
            // No continuation is to be executed, register the plain
            // action and the local-virtual address.
            action_->schedule_thread(std::move(cont_), std::move(data_.dest_),
                lva, std::size_t(-1));
        }
        else
        {
            // This parcel carries a continuation, register a wrapper
            // which first executes the original thread function as
            // required by the action and triggers the continuations
            // afterwards.
            action_->schedule_thread(std::move(data_.dest_), lva, std::size_t(-1));
        }
    }

    void parcel::load_data(serialization::input_archive & ar)
    {
        using hpx::serialization::detail::polymorphic_id_factory;
        ar >> data_;
        ar >> cont_;
        std::uint32_t id;
        ar >> id;

#if !defined(HPX_DEBUG)
        action_.reset(polymorphic_id_factory::create<actions::base_action>(id));
#else
        std::string name;
        ar >> name;
        action_.reset(
            polymorphic_id_factory::create<actions::base_action>(id, &name));
#endif
    }

    void parcel::serialize(serialization::input_archive & ar, unsigned)
    {
        load_data(ar);
        ar >> *action_;
    }

    void parcel::serialize(serialization::output_archive & ar, unsigned)
    {
        using hpx::serialization::detail::polymorphic_id_factory;
        using hpx::serialization::access;

        ar & data_;
        ar & cont_;
#if !defined(HPX_DEBUG)
        const std::uint32_t id =
            polymorphic_id_factory::get_id(
                access::get_name(action_.get()));
        ar << id;
#else
        std::string const name(access::get_name(action_.get()));
        const std::uint32_t id =
            polymorphic_id_factory::get_id(name);
        ar << id;
        ar << name;
#endif
        ar << *action_;
    }

    ///////////////////////////////////////////////////////////////////////////
    std::ostream& operator<< (std::ostream& os, parcel const& p)
    {
        os << "(" << p.data_.dest_ << ":" << p.data_.addr_ << ":";
        os << p.action_->get_action_name() << ")";

        return os;
    }

    std::string dump_parcel(parcel const& p)
    {
        std::ostringstream os;
        os << p;
        return os.str();
    }
}}

