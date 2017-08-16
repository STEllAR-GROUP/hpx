//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime.hpp>
#include <hpx/runtime/applier/applier.hpp>
#if defined(HPX_DEBUG)
#include <hpx/runtime/components/component_type.hpp>
#endif
#include <hpx/runtime/components/runtime_support.hpp>
#include <hpx/runtime/actions/base_action.hpp>
#include <hpx/runtime/actions/detail/action_factory.hpp>
#include <hpx/runtime/agas/addressing_service.hpp>
#include <hpx/runtime/parcelset/parcel.hpp>
#include <hpx/runtime/parcelset/parcelhandler.hpp>
#include <hpx/runtime/parcelset/detail/parcel_route_handler.hpp>
#include <hpx/runtime/serialization/access.hpp>
#include <hpx/runtime/serialization/input_archive.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>
#include <hpx/runtime/serialization/detail/polymorphic_id_factory.hpp>
#include <hpx/util/apex.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/util/itt_notify.hpp>

#include <hpx/util/atomic_count.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset
{
    namespace detail
    {
        parcel_data::parcel_data() :
#if defined(HPX_HAVE_PARCEL_PROFILING)
            start_time_(0),
            creation_time_(util::high_resolution_timer::now()),
#endif
            source_id_(naming::invalid_gid),
            dest_(naming::invalid_gid),
            has_continuation_(false)
        {}

        parcel_data::parcel_data(naming::gid_type&& dest, naming::address&& addr,
            bool has_continuation) :
#if defined(HPX_HAVE_PARCEL_PROFILING)
            start_time_(0),
            creation_time_(util::high_resolution_timer::now()),
#endif
            source_id_(naming::invalid_gid),
            dest_(std::move(dest)),
            addr_(std::move(addr)),
            has_continuation_(has_continuation)
        {}

        parcel_data::parcel_data(parcel_data && rhs) :
#if defined(HPX_HAVE_PARCEL_PROFILING)
            parcel_id_(std::move(rhs.parcel_id_)),
            start_time_(rhs.start_time_),
            creation_time_(rhs.creation_time_),
#endif
            source_id_(std::move(rhs.source_id_)),
            dest_(std::move(rhs.dest_)),
            addr_(std::move(rhs.addr_)),
            has_continuation_(rhs.has_continuation_)
        {
#if defined(HPX_HAVE_PARCEL_PROFILING)
            rhs.parcel_id_ = naming::invalid_gid;
            rhs.start_time_ = 0;
            rhs.creation_time_ = 0;
#endif
            rhs.source_id_ = naming::invalid_gid;
            rhs.dest_ = naming::invalid_gid;
            rhs.addr_ = naming::address();
        }

        parcel_data& parcel_data::operator=(parcel_data && rhs)
        {
#if defined(HPX_HAVE_PARCEL_PROFILING)
            parcel_id_ = std::move(rhs.parcel_id_);
            start_time_ = rhs.start_time_;
            creation_time_ = rhs.creation_time_;
#endif
            source_id_ = std::move(rhs.source_id_);
            dest_ = std::move(rhs.dest_);
            addr_ = std::move(rhs.addr_);
            has_continuation_ = rhs.has_continuation_;

#if defined(HPX_HAVE_PARCEL_PROFILING)
            rhs.parcel_id_ = naming::invalid_gid;
            rhs.start_time_ = 0;
            rhs.creation_time_ = 0;
#endif
            rhs.source_id_ = naming::invalid_gid;
            rhs.dest_ = naming::invalid_gid;
            rhs.addr_ = naming::address();
            return *this;
        }

        template <typename Archive>
        void parcel_data::serialize(Archive &ar, unsigned)
        {
#if defined(HPX_HAVE_PARCEL_PROFILING)
            ar & parcel_id_;
            ar & start_time_;
            ar & creation_time_;
#endif
            ar & source_id_;
            ar & dest_;
            ar & addr_;

            ar & has_continuation_;
        }

        template void parcel_data::serialize(
            hpx::serialization::input_archive&, unsigned);
        template void parcel_data::serialize(
            hpx::serialization::output_archive&, unsigned);
    }
}}

namespace hpx { namespace parcelset
{
#if defined(HPX_DEBUG)
    bool parcel::is_valid() const
    {
        // empty parcels are always valid
#if defined(HPX_HAVE_PARCEL_PROFILING)
        if (0.0 == data_.creation_time_) //-V550
            return true;
#endif

        // verify target destination
        if (data_.dest_ && data_.addr_.locality_)
        {
            // if we have a destination we need an action as well
            if (!action_)
                return false;
        }

        // verify that the action targets the correct type
        if (action_ && data_.addr_.type_ != components::component_invalid)
        {
            int type = action_->get_component_type();
            if (!components::types_are_compatible(type, data_.addr_.type_))
            {
                return false;
            }
        }

        return true;
    }
#endif

    parcel::parcel()
    {}

    parcel::~parcel()
    {}

    parcel::parcel(
        naming::gid_type&& dest,
        naming::address&& addr,
        std::unique_ptr<actions::base_action> act
    )
      : data_(std::move(dest), std::move(addr), act->has_continuation()),
        action_(std::move(act)),
        size_(0)
    {
//             HPX_ASSERT(is_valid());
    }

    parcel::parcel(parcel && other)
      : data_(std::move(other.data_)),
        action_(std::move(other.action_)),
        split_gids_(std::move(other.split_gids_)),
        size_(other.size_),
        num_chunks_(other.num_chunks_)
    {
        HPX_ASSERT(is_valid());
    }

    parcel &parcel::operator=(parcel && other)
    {
        data_ = std::move(other.data_);
        action_ = std::move(other.action_);
        split_gids_ = std::move(other.split_gids_);
        size_ = other.size_;
        num_chunks_ = other.num_chunks_;

        other.reset();

        HPX_ASSERT(is_valid());
        return *this;
    }

    void parcel::reset()
    {
        data_ = detail::parcel_data();
        action_.reset();
    }

    actions::base_action *parcel::get_action() const
    {
        HPX_ASSERT(action_.get());
        return action_.get();
    }

    naming::id_type parcel::source_id() const
    {
        return naming::id_type(data_.source_id_, naming::id_type::unmanaged);
    }

    void parcel::set_source_id(naming::id_type const & source_id)
    {
        if (source_id != naming::invalid_id)
        {
            data_.source_id_ = source_id.get_gid();
        }
    }

    void parcel::set_destination_id(naming::gid_type&& dest)
    {
        data_.dest_ = dest;
        HPX_ASSERT(is_valid());
    }

    naming::gid_type const& parcel::destination() const
    {
        HPX_ASSERT(is_valid());
        return data_.dest_;
    }

    naming::address const& parcel::addr() const
    {
        return data_.addr_;
    }

    naming::address& parcel::addr()
    {
        return data_.addr_;
    }

    std::uint32_t parcel::destination_locality_id() const
    {
        return naming::get_locality_id_from_gid(destination_locality());
    }

    naming::gid_type const& parcel::destination_locality() const
    {
        return addr().locality_;
    }

    double parcel::start_time() const
    {
#if defined(HPX_HAVE_PARCEL_PROFILING)
        return data_.start_time_;
#else
        return 0.0;
#endif
    }

    void parcel::set_start_time(double time)
    {
#if defined(HPX_HAVE_PARCEL_PROFILING)
        data_.start_time_ = time;
#endif
    }

    double parcel::creation_time() const
    {
#if defined(HPX_HAVE_PARCEL_PROFILING)
        return data_.creation_time_;
#else
        return 0.0;
#endif
    }

    threads::thread_priority parcel::get_thread_priority() const
    {
        return action_->get_thread_priority();
    }

#if defined(HPX_HAVE_PARCEL_PROFILING)
    naming::gid_type const parcel::parcel_id() const
    {
        return data_.parcel_id_;
    }

    naming::gid_type & parcel::parcel_id()
    {
        return data_.parcel_id_;
    }
#endif

    serialization::binary_filter* parcel::get_serialization_filter() const
    {
        return action_->get_serialization_filter(*this);
    }

    policies::message_handler* parcel::get_message_handler(
        parcelset::parcelhandler* ph, locality const& loc) const
    {
        return action_->get_message_handler(ph, loc, *this);
    }

    bool parcel::does_termination_detection() const
    {
        return action_ ? action_->does_termination_detection() : false;
    }

    parcel::split_gids_type& parcel::split_gids() const
    {
        return const_cast<split_gids_type&>(split_gids_);
    }

    void parcel::set_split_gids(parcel::split_gids_type&& split_gids)
    {
        split_gids_ = std::move(split_gids);
    }

    std::size_t const& parcel::num_chunks() const
    {
        return num_chunks_;
    }

    std::size_t & parcel::num_chunks()
    {
        return num_chunks_;
    }

    std::size_t const& parcel::size() const
    {
        return size_;
    }

    std::size_t & parcel::size()
    {
        return size_;
    }

    ///////////////////////////////////////////////////////////////////////////
    // generate unique parcel id
    naming::gid_type parcel::generate_unique_id(
        std::uint32_t locality_id_default)
    {
        static hpx::util::atomic_count id(0);

        error_code ec(lightweight);        // ignore all errors
        std::uint32_t locality_id = hpx::get_locality_id(ec);
        if (locality_id == naming::invalid_locality_id)
            locality_id = locality_id_default;

        naming::gid_type result = naming::get_gid_from_locality_id(locality_id);
        result.set_lsb(++id);
        return result;
    }

    std::pair<naming::address_type, naming::component_type>
        parcel::determine_lva()
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

        return std::make_pair(lva, comptype);
    }

    bool parcel::load_schedule(serialization::input_archive & ar,
        std::size_t num_thread, bool& deferred_schedule)
    {
        load_data(ar);

        // make sure this parcel destination matches the proper locality
        HPX_ASSERT(destination_locality() == data_.addr_.locality_);

        std::pair<naming::address_type, naming::component_type> p = determine_lva();

        // make sure the target has not been migrated away
        auto r = action_->was_object_migrated(data_.dest_, p.first);
        if (r.first)
        {
            // If the object was migrated, just load the action and return.
            action_->load(ar);
            return true;
        }

        // continuation support, this is handled in the transfer action
        action_->load_schedule(ar, std::move(data_.dest_), p.first, p.second,
            num_thread, deferred_schedule);

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
        static util::itt::event parcel_recv("recv_parcel");
        util::itt::event_tick(parcel_recv);
#endif

#if defined(HPX_HAVE_APEX) && defined(HPX_HAVE_PARCEL_PROFILING)
        // tell APEX about the received parcel
        apex::recv(data_.parcel_id_.get_lsb(), size_,
            naming::get_locality_id_from_gid(data_.source_id_),
            reinterpret_cast<std::uint64_t>(action_->get_parent_thread_id()));
#endif

        return false;
    }

    void parcel::schedule_action(std::size_t num_thread)
    {
        // make sure this parcel destination matches the proper locality
        HPX_ASSERT(destination_locality() == data_.addr_.locality_);

        std::pair<naming::address_type, naming::component_type> p = determine_lva();

        // make sure the target has not been migrated away
        auto r = action_->was_object_migrated(data_.dest_, p.first);
        if (r.first)
        {
            // If the object was migrated, just route.
            naming::resolver_client& client = hpx::naming::get_agas_client();
            client.route(
                std::move(*this),
                &detail::parcel_route_handler,
                threads::thread_priority_normal);
            return;
        }

        // dispatch action, register work item either with or without
        // continuation support, this is handled in the transfer action
        action_->schedule_thread(std::move(data_.dest_), p.first, p.second,
            num_thread);
    }

    void parcel::load_data(serialization::input_archive & ar)
    {
        using hpx::actions::detail::action_registry;
        ar >> data_;
        std::uint32_t id;
        ar >> id;

#if !defined(HPX_DEBUG)
        action_.reset(action_registry::create(id, data_.has_continuation_));
#else
        std::string name;
        ar >> name;
        action_.reset(action_registry::create(id, data_.has_continuation_, &name));
#endif
    }

    void parcel::serialize(serialization::input_archive & ar, unsigned)
    {
        load_data(ar);
        action_->load(ar);
    }

    void parcel::serialize(serialization::output_archive & ar, unsigned)
    {
        using hpx::serialization::access;

        ar & data_;
#if !defined(HPX_DEBUG)
        const std::uint32_t id = action_->get_action_id();
        ar << id;
#else
        std::string const name(action_->get_action_name());
        const std::uint32_t id = action_->get_action_id();
        ar << id;
        ar << name;
#endif
        action_->save(ar);
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

