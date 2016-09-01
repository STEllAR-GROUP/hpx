//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2015-2016 Thomas Heller
//  Copyright (c) 2007 Richard D Guidry Jr
//  Copyright (c) 2007 Alexandre (aka Alex) TABBAL
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_PARCEL_MAR_26_2008_1051AM)
#define HPX_PARCELSET_PARCEL_MAR_26_2008_1051AM

#include <hpx/config.hpp>
#include <hpx/runtime/actions/transfer_action.hpp>
#if defined(HPX_DEBUG)
#include <hpx/runtime/components/component_type.hpp>
#endif
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/parcelset_fwd.hpp>
#include <hpx/runtime/serialization/input_archive.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>
#include <hpx/traits/is_bitwise_serializable.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/atomic_count.hpp>
#include <hpx/util/high_resolution_timer.hpp>

#include <boost/intrusive_ptr.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <unordered_map>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace serialization
{
    struct binary_filter;
}}
namespace hpx { namespace parcelset { namespace policies
{
    struct message_handler;
}}}

namespace hpx { namespace parcelset
{
    class HPX_EXPORT parcel
    {
    private:
        HPX_MOVABLE_ONLY(parcel);

    private:

        typedef
            std::unordered_map<naming::gid_type, naming::gid_type>
            splitted_gids_type;

#if defined(HPX_DEBUG)
        bool is_valid() const
        {
            // empty parcels are always valid
            if (0 == data_.creation_time_) //-V550
                return true;

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
#else
        // Only used in debug mode.
        bool is_valid() const
        {
            return true;
        }
#endif

    public:
        struct data
        {
        private:
            HPX_MOVABLE_ONLY(data);

        public:
            data()
              : start_time_(0),
                creation_time_(util::high_resolution_timer::now()),
                source_id_(naming::invalid_gid),
                dest_(naming::invalid_gid)
            {}

            data(naming::gid_type&& dest, naming::address&& addr)
              : start_time_(0),
                creation_time_(util::high_resolution_timer::now()),
                source_id_(naming::invalid_gid),
                dest_(std::move(dest)),
                addr_(std::move(addr))
            {}

            data(data && rhs)
              : start_time_(rhs.start_time_),
                creation_time_(rhs.creation_time_),
                source_id_(std::move(rhs.source_id_)),
                dest_(std::move(rhs.dest_)),
                addr_(std::move(rhs.addr_))
            {
                rhs.start_time_ = 0;
                rhs.creation_time_ = 0;
                rhs.creation_time_ = 0;
                rhs.source_id_ = naming::invalid_gid;
                rhs.dest_ = naming::invalid_gid;
                rhs.addr_ = naming::address();
            }

            data& operator=(data && rhs)
            {
                start_time_ = rhs.start_time_;
                creation_time_ = rhs.creation_time_;
                source_id_ = std::move(rhs.source_id_);
                dest_ = std::move(rhs.dest_);
                addr_ = std::move(rhs.addr_);

                rhs.start_time_ = 0;
                rhs.creation_time_ = 0;
                rhs.source_id_ = naming::invalid_gid;
                rhs.dest_ = naming::invalid_gid;
                rhs.addr_ = naming::address();
                return *this;
            }

            template <typename Archive>
            void serialize(Archive &ar, unsigned)
            {
                ar & start_time_;
                ar & creation_time_;
                ar & source_id_;
                ar & dest_;
                ar & addr_;
            }

            double start_time_;
            double creation_time_;

            naming::gid_type source_id_;
            naming::gid_type dest_;
            naming::address addr_;
        };

        parcel() {}

    private:
        parcel(
            naming::gid_type&& dest,
            naming::address&& addr,
            std::unique_ptr<actions::continuation> cont,
            std::unique_ptr<actions::base_action> act,
            bool action_args_bitwise = false
        )
          : data_(std::move(dest), std::move(addr)),
            cont_(std::move(cont)),
            action_(std::move(act)),
            size_(0),
            action_args_bitwise_(action_args_bitwise)
        {
//             HPX_ASSERT(is_valid());
        }
        friend struct detail::create_parcel;
    public:

        parcel(parcel && other)
          : data_(std::move(other.data_)),
            cont_(std::move(other.cont_)),
            action_(std::move(other.action_)),
            splitted_gids_(std::move(other.splitted_gids_)),
            size_(other.size_),
            num_chunks_(other.num_chunks_),
            action_args_bitwise_(other.action_args_bitwise_)
        {
            HPX_ASSERT(is_valid());
        }

        parcel &operator=(parcel && other)
        {
            data_ = std::move(other.data_);
            cont_ = std::move(other.cont_);
            action_ = std::move(other.action_);
            splitted_gids_ = std::move(other.splitted_gids_);
            size_ = other.size_;
            num_chunks_ = other.num_chunks_;
            action_args_bitwise_ = other.action_args_bitwise_;

            other.reset();

            HPX_ASSERT(is_valid());
            return *this;
        }

        void reset()
        {
            data_ = data();
            cont_.reset();
            action_.reset();
        }

        actions::base_action *get_action() const
        {
            HPX_ASSERT(action_.get());
            return action_.get();
        }

        std::unique_ptr<actions::continuation> get_continuation()
        {
            return std::move(cont_);
        }
        void set_continuation(std::unique_ptr<actions::continuation> cont)
        {
            cont_ = std::move(cont);
        }

        naming::id_type source_id() const
        {
            return naming::id_type(data_.source_id_, naming::id_type::unmanaged);
        }

        void set_source_id(naming::id_type const & source_id)
        {
            if (source_id != naming::invalid_id)
            {
                data_.source_id_ = source_id.get_gid();
            }
        }

        void set_destination_id(naming::gid_type&& dest)
        {
            data_.dest_ = dest;
            HPX_ASSERT(is_valid());
        }

        naming::gid_type const& destination() const
        {
            HPX_ASSERT(is_valid());
            return data_.dest_;
        }

        naming::address const& addr() const
        {
            return data_.addr_;
        }

        naming::address& addr()
        {
            return data_.addr_;
        }

        std::uint32_t destination_locality_id() const
        {
            return naming::get_locality_id_from_gid(destination_locality());
        }

        naming::gid_type const& destination_locality() const
        {
            return addr().locality_;
        }

        double start_time() const
        {
            return data_.start_time_;
        }

        void set_start_time(double time)
        {
            data_.start_time_ = time;
        }

        double creation_time() const
        {
            return data_.creation_time_;
        }

        threads::thread_priority get_thread_priority() const
        {
            return action_->get_thread_priority();
        }

        serialization::binary_filter* get_serialization_filter() const
        {
            return action_->get_serialization_filter(*this);
        }

        policies::message_handler* get_message_handler(
            parcelset::parcelhandler* ph, locality const& loc) const
        {
            return action_->get_message_handler(ph, loc, *this);
        }

        bool does_termination_detection() const
        {
            return action_ ? action_->does_termination_detection() : false;
        }

        splitted_gids_type& splitted_gids() const
        {
            return const_cast<splitted_gids_type&>(splitted_gids_);
        }

        void set_splitted_gids(splitted_gids_type&& splitted_gids)
        {
            splitted_gids_ = std::move(splitted_gids);
        }

        std::size_t const& num_chunks() const
        {
            return num_chunks_;
        }

        std::size_t & num_chunks()
        {
            return num_chunks_;
        }

        bool action_args_bitwise() const
        {
            return action_args_bitwise_;
        }

        std::size_t const& size() const
        {
            return size_;
        }

        std::size_t & size()
        {
            return size_;
        }

        void schedule_action();

        void load_schedule(serialization::input_archive & ar,
            std::size_t num_thread);

    private:
        friend std::ostream& operator<< (std::ostream& os, parcel const& req);

        // serialization support
        friend class hpx::serialization::access;
        void serialize(serialization::input_archive & ar, unsigned);
        void serialize(serialization::output_archive & ar, unsigned);

        naming::address_type determine_lva();

        data data_;
        std::unique_ptr<actions::continuation> cont_;
        std::unique_ptr<actions::base_action> action_;

        splitted_gids_type splitted_gids_;
        std::size_t size_;
        std::size_t num_chunks_;
        bool action_args_bitwise_;
    };

    HPX_EXPORT std::string dump_parcel(parcel const& p);
}}

HPX_IS_BITWISE_SERIALIZABLE(hpx::parcelset::parcel::data)

#include <hpx/config/warnings_suffix.hpp>

#endif
