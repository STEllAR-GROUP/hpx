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
#include <hpx/exception.hpp>
#include <hpx/runtime/actions/transfer_action.hpp>
#if defined(HPX_DEBUG)
#include <hpx/runtime/components/component_type.hpp>
#endif
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/serialization/input_archive.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>
#include <hpx/traits/is_bitwise_serializable.hpp>
#include <hpx/traits/is_action.hpp>
#include <hpx/traits/is_continuation.hpp>
#include <hpx/util/assert.hpp>

#include <boost/intrusive_ptr.hpp>
#include <boost/detail/atomic_count.hpp>

#include <memory>

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
        HPX_MOVABLE_BUT_NOT_COPYABLE(parcel);
    public:

        struct data
        {
            data()
              : start_time_(0),
                creation_time_(0),
                has_source_id_(false)
            {}

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
            naming::id_type const& dests,
            naming::address const& addrs,
            Action,
            Args &&... args
        )
#if defined(HPX_SUPPORT_MULTIPLE_PARCEL_DESTINATIONS)
          : dests_(1, dests),
            addrs_(1, addrs),
#else
          : dests_(dests),
            addrs_(addrs),
#endif
            action_(new actions::transfer_action<Action>(std::forward<Args>(args)...))
        {
            HPX_ASSERT(components::types_are_compatible(
                action_->get_component_type(), addrs.type_));
        }

#if defined(HPX_SUPPORT_MULTIPLE_PARCEL_DESTINATIONS)
        template <
            typename Action
          , typename...Args
          , typename Enable =
                typename std::enable_if<traits::is_action<Action>::value>::type
          >
        parcel(
            std::vector<naming::id_type> const& dests,
            std::vector<naming::address> const& addrs,
            Action,
            Args &&... args
        )
          : dests_(dests),
            addrs_(addrs),
            action_(new actions::transfer_action<Action>(std::forward<Args>(args)...))
        {
            HPX_ASSERT(dests_.size() == addrs_.size());
#if defined(HPX_DEBUG)
            if (!dests_.empty() && addrs_[0].locality_)
            {
                // all destinations have to be on the same locality
                naming::gid_type dest = destination_locality();
                for (std::size_t i = 1; i != addrs_.size(); ++i)
                {
                    HPX_ASSERT(dest == addrs_[i].locality_);
                }

                // all destination component types are properly matched
                int comptype = action_->get_component_type();
                for (std::size_t i = 0; i != addrs_.size(); ++i)
                {
                    HPX_ASSERT(components::types_are_compatible(
                        comptype, addrs[i].type_));
                }
                HPX_UNUSED(comptype);
            }
#endif
        }
#endif

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
            naming::id_type const& dests,
            naming::address const& addrs,
            Continuation && cont,
            Action,
            Args &&... args
        )
#if defined(HPX_SUPPORT_MULTIPLE_PARCEL_DESTINATIONS)
          : dests_(1, dests),
            addrs_(1, addrs),
#else
          : dests_(dests),
            addrs_(addrs),
#endif
            cont_(
                new typename util::decay<Continuation>
                    ::type(std::forward<Continuation>(cont))
            ),
            action_(new actions::transfer_action<Action>(std::forward<Args>(args)...))
        {
            HPX_ASSERT(components::types_are_compatible(
                action_->get_component_type(), this->addrs()[0].type_
            ));
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
            naming::id_type const& dests,
            naming::address const& addrs,
            std::unique_ptr<actions::continuation> cont,
            Action,
            Args &&... args
        )
#if defined(HPX_SUPPORT_MULTIPLE_PARCEL_DESTINATIONS)
          : dests_(1, dests),
            addrs_(1, addrs),
#else
          : dests_(dests),
            addrs_(addrs),
#endif
            cont_(std::move(cont)),
            action_(new actions::transfer_action<Action>(std::forward<Args>(args)...))
        {
            HPX_ASSERT(components::types_are_compatible(
                action_->get_component_type(), this->addrs()[0].type_
            ));
        }

        parcel(parcel && other)
          : data_(std::move(other.data_)),
            source_id_(std::move(other.source_id_)),
            dests_(std::move(other.dests_)),
            addrs_(std::move(other.addrs_)),
            cont_(std::move(other.cont_)),
            action_(std::move(other.action_))
        {}

        parcel &operator=(parcel && other)
        {
            data_ = std::move(other.data_);
            source_id_ = std::move(other.source_id_);
            dests_ = std::move(other.dests_);
            addrs_ = std::move(other.addrs_);
            cont_ = std::move(other.cont_);
            action_ = std::move(other.action_);
            return *this;
        }

        void reset()
        {
            data_ = data();
            source_id_ = hpx::naming::invalid_id;
#if defined(HPX_SUPPORT_MULTIPLE_PARCEL_DESTINATIONS)
            dests_.clear();
            addrs_.clear();
#else
            dests_ = hpx::naming::invalid_id;
            addrs_ = hpx::naming::address();
#endif
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

        naming::id_type const& source_id() const
        {
            return source_id_;
        }

        void set_source_id(naming::id_type const & source_id)
        {
            if(source_id != naming::invalid_id)
            {
                source_id_ = source_id;
                data_.has_source_id_ = true;
            }
        }

        std::size_t size() const
        {
#if defined(HPX_SUPPORT_MULTIPLE_PARCEL_DESTINATIONS)
            return dests_.size();
#else
            return 1ul;
#endif
        }

        naming::id_type const* destinations() const
        {

#if defined(HPX_SUPPORT_MULTIPLE_PARCEL_DESTINATIONS)
            return dests_.data();
#else
            return &dests_;
#endif
        }

        naming::address const* addrs() const
        {
#if defined(HPX_SUPPORT_MULTIPLE_PARCEL_DESTINATIONS)
            return addrs_.data();
#else
            return &addrs_;
#endif
        }

        naming::address * addrs()
        {
#if defined(HPX_SUPPORT_MULTIPLE_PARCEL_DESTINATIONS)
            return addrs_.data();
#else
            return &addrs_;
#endif
        }

        boost::uint32_t destination_locality_id() const
        {
            return naming::get_locality_id_from_id(destinations()[0]);
        }

        naming::gid_type const& destination_locality() const
        {
            return addrs()[0].locality_;
        }

        double const start_time() const
        {
            return data_.start_time_;
        }

        void set_start_time(double time)
        {
            data_.start_time_ = time;
            if(std::abs(data_.creation_time_) < 1e-10)
            {
                data_.creation_time_ = time;
            }
        }

        double const creation_time() const
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

#if defined(HPX_SUPPORT_MULTIPLE_PARCEL_DESTINATIONS)
        std::vector<naming::id_type> dests_;
        std::vector<naming::address> addrs_;
#else
        naming::id_type dests_;
        naming::address addrs_;
#endif
        std::unique_ptr<actions::continuation> cont_;
        std::unique_ptr<actions::base_action> action_;
    };

    HPX_EXPORT std::string dump_parcel(parcel const& p);
}}

namespace hpx { namespace traits
{
    template <>
    struct is_bitwise_serializable<
            hpx::parcelset::parcel::data>
       : boost::mpl::true_
    {};
}}

#include <hpx/config/warnings_suffix.hpp>

#endif
