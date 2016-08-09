//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2015 Thomas Heller
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
#include <hpx/runtime/serialization/input_archive.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>
#include <hpx/traits/is_action.hpp>
#include <hpx/traits/is_bitwise_serializable.hpp>
#include <hpx/traits/is_continuation.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/atomic_count.hpp>
#include <hpx/util/high_resolution_timer.hpp>

#include <boost/intrusive_ptr.hpp>

#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

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
#if defined(HPX_DEBUG)
        bool is_valid() const
        {
            // empty parcels are always valid
            if (0 == data_.creation_time_) //-V550
                return true;

            if (data_.has_source_id_ && !source_id_)
                return false;

            // verify target destination
            if (dest_ && addr_.locality_)
            {
                // if we have a destination we need an action as well
                if (!action_)
                    return false;
            }

            // verify that the action targets the correct type
            if (action_ && addr_.type_ != components::component_invalid)
            {
                int type = action_->get_component_type();
                if (!components::types_are_compatible(type, addr_.type_))
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
                has_source_id_(false)
            {}

            data(data && rhs)
              : parcel_id_(rhs.parcel_id_),
                start_time_(rhs.start_time_),
                creation_time_(rhs.creation_time_),
                has_source_id_(rhs.has_source_id_)
            {
                rhs.parcel_id_ = naming::invalid_gid;
                rhs.start_time_ = 0;
                rhs.creation_time_ = 0;
                rhs.has_source_id_ = false;
            }

            data& operator=(data && rhs)
            {
                parcel_id_ = rhs.parcel_id_;
                start_time_ = rhs.start_time_;
                creation_time_ = rhs.creation_time_;
                has_source_id_ = rhs.has_source_id_;

                rhs.parcel_id_ = naming::invalid_gid;
                rhs.start_time_ = 0;
                rhs.creation_time_ = 0;
                rhs.has_source_id_ = false;
                return *this;
            }

            template <typename Archive>
            void serialize(Archive &ar, unsigned)
            {
                ar & parcel_id_;
                ar & start_time_;
                ar & creation_time_;
                ar & has_source_id_;
            }

            naming::gid_type parcel_id_;
            double start_time_;
            double creation_time_;

            bool has_source_id_;
        };

        parcel() {}

        template <
            typename Action
          , typename...Args
          , typename Enable =
                typename std::enable_if<traits::is_action<Action>::value>::type
          >
        parcel(
            naming::id_type const& dest,
            naming::address const& addr,
            Action,
            Args &&... args
        )
          : dest_(dest),
            addr_(addr),
            action_(new actions::transfer_action<Action>(std::forward<Args>(args)...))
        {
            HPX_ASSERT(is_valid());
        }

        template <
            typename Continuation
          , typename Action
          , typename...Args
          , typename Enable
                = typename std::enable_if<
                    traits::is_action<Action>::value
                 && traits::is_continuation<Continuation>::value
                 && !std::is_same<
                        Continuation
                      , std::unique_ptr<actions::continuation>
                    >::value
                >::type
          >
        parcel(
            naming::id_type const& dest,
            naming::address const& addr,
            Continuation && cont,
            Action,
            Args &&... args
        )
          : dest_(dest),
            addr_(addr),
            cont_(
                new typename util::decay<Continuation>
                    ::type(std::forward<Continuation>(cont))
            ),
            action_(new actions::transfer_action<Action>(std::forward<Args>(args)...))
        {
            HPX_ASSERT(is_valid());
        }

        template <
            typename Action
          , typename...Args
          , typename Enable
                = typename std::enable_if<
                    traits::is_action<Action>::value
                >::type
          >
        parcel(
            naming::id_type const& dest,
            naming::address const& addr,
            std::unique_ptr<actions::continuation> cont,
            Action,
            Args &&... args
        )
          : dest_(dest),
            addr_(addr),
            cont_(std::move(cont)),
            action_(new actions::transfer_action<Action>(std::forward<Args>(args)...))
        {
            HPX_ASSERT(is_valid());
        }

        parcel(parcel && other)
          : data_(std::move(other.data_)),
            source_id_(std::move(other.source_id_)),
            dest_(std::move(other.dest_)),
            addr_(std::move(other.addr_)),
            cont_(std::move(other.cont_)),
            action_(std::move(other.action_))
        {
            HPX_ASSERT(is_valid());
        }

        parcel &operator=(parcel && other)
        {
            data_ = std::move(other.data_);
            source_id_ = std::move(other.source_id_);
            dest_ = std::move(other.dest_);
            addr_ = std::move(other.addr_);
            cont_ = std::move(other.cont_);
            action_ = std::move(other.action_);

            other.reset();

            HPX_ASSERT(is_valid());
            return *this;
        }

        void reset()
        {
            data_ = data();
            source_id_ = hpx::naming::invalid_id;
            dest_ = hpx::naming::invalid_id;
            addr_ = hpx::naming::address();
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

        naming::id_type const& source_id() const
        {
            return source_id_;
        }

        void set_source_id(naming::id_type const & source_id)
        {
            if (source_id != naming::invalid_id)
            {
                source_id_ = source_id;
                data_.has_source_id_ = true;
            }
        }

        naming::id_type const& destination() const
        {

            return dest_;
        }

        naming::address const& addr() const
        {
            return addr_;
        }

        naming::address& addr()
        {
            return addr_;
        }

        boost::uint32_t destination_locality_id() const
        {
            return naming::get_locality_id_from_id(destination());
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

        naming::gid_type const parcel_id() const
        {
            return data_.parcel_id_;
        }

        naming::gid_type & parcel_id()
        {
            return data_.parcel_id_;
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

        // generate unique parcel id
        static naming::gid_type generate_unique_id(
            boost::uint32_t locality_id = naming::invalid_locality_id);

    private:
        friend std::ostream& operator<< (std::ostream& os, parcel const& req);

        // serialization support
        friend class hpx::serialization::access;

        void serialize(serialization::input_archive & ar, unsigned);
        void serialize(serialization::output_archive & ar, unsigned);

        data data_;
        naming::id_type source_id_;

        naming::id_type dest_;
        naming::address addr_;
        std::unique_ptr<actions::continuation> cont_;
        std::unique_ptr<actions::base_action> action_;
    };

    HPX_EXPORT std::string dump_parcel(parcel const& p);
}}

HPX_IS_BITWISE_SERIALIZABLE(hpx::parcelset::parcel::data)

#include <hpx/config/warnings_suffix.hpp>

#endif
