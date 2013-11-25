//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2007 Richard D Guidry Jr
//  Copyright (c) 2007 Alexandre (aka Alex) TABBAL
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_PARCEL_MAR_26_2008_1051AM)
#define HPX_PARCELSET_PARCEL_MAR_26_2008_1051AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/parcelset/policies/message_handler.hpp>
#include <hpx/util/binary_filter.hpp>

#include <boost/serialization/split_member.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/is_bitwise_serializable.hpp>

#include <boost/intrusive_ptr.hpp>
#include <hpx/assert.hpp>
#include <boost/detail/atomic_count.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
//  parcel serialization format version
#define HPX_PARCEL_VERSION 0x80

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset
{
    class HPX_EXPORT parcel;

    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////////
        class HPX_EXPORT parcel_data
        {
        public:
            parcel_data()
              : count_(0)
            {}

            parcel_data(actions::base_action* act)
              : count_(0), action_(act)
            {}

            parcel_data(actions::action_type act)
              : count_(0), action_(act)
            {}

            parcel_data(actions::base_action* act,
                   actions::continuation* do_after)
              : count_(0),
                action_(act), continuation_(do_after)
            {}

            parcel_data(actions::base_action* act,
                    actions::continuation_type do_after)
              : count_(0),
                action_(act), continuation_(do_after)
            {}

            virtual ~parcel_data() {}

            virtual bool is_multi_destination() const = 0;
            virtual std::size_t size() const = 0;

            ///
            virtual void set_start_time(double starttime) = 0;
            virtual double get_start_time() const = 0;
            virtual double get_creation_time() const = 0;

            /// get and set the destination id
            virtual naming::id_type* get_destinations() = 0;
            virtual naming::id_type const* get_destinations() const = 0;
            virtual void set_destination(naming::id_type const& dest) = 0;
#if defined(HPX_SUPPORT_MULTIPLE_PARCEL_DESTINATIONS)
            virtual void set_destinations(std::vector<naming::id_type> const& dests) = 0;
#endif
            /// get and set the destination address
            virtual naming::address* get_destination_addrs() = 0;
            virtual naming::address const* get_destination_addrs() const = 0;
            virtual void set_destination_addr(naming::address const& addr) = 0;
#if defined(HPX_SUPPORT_MULTIPLE_PARCEL_DESTINATIONS)
            virtual void set_destination_addrs(std::vector<naming::address> const& addrs) = 0;
#endif
            virtual naming::locality const& get_destination_locality() const = 0;

            virtual naming::gid_type get_parcel_id() const = 0;
            virtual void set_parcel_id(naming::gid_type const& id) = 0;

            virtual std::size_t get_type_size() const = 0;

            // default copy constructor is ok
            // default assignment operator is ok

            /// get and set the source locality/component id
            naming::id_type& get_source()
            {
                return source_id_;
            }
            naming::id_type const& get_source() const
            {
                return source_id_;
            }
            void set_source(naming::id_type const& source_id)
            {
                source_id_ = source_id;
            }

            actions::action_type get_action() const
            {
                return action_;
            }

            actions::continuation_type get_continuation() const
            {
                return continuation_;
            }

            threads::thread_priority get_thread_priority() const
            {
                HPX_ASSERT(action_);
                return action_->get_thread_priority();
            }

            util::binary_filter* get_serialization_filter(parcelset::parcel const& p) const
            {
                return action_->get_serialization_filter(p);
            }

            policies::message_handler* get_message_handler(
                parcelset::parcelhandler* ph, naming::locality const& loc,
                parcelset::connection_type t, parcelset::parcel const& p) const
            {
                return action_->get_message_handler(ph, loc, t, p);
            }

        protected:
            template <typename Archive>
            void save(Archive& ar, bool has_source_id, bool has_continuation) const;

            template <typename Archive>
            void load(Archive& ar, bool has_source_id, bool has_continuation);

        private:
            friend void intrusive_ptr_add_ref(parcel_data* p);
            friend void intrusive_ptr_release(parcel_data* p);

            boost::detail::atomic_count count_;

        protected:
            naming::id_type source_id_;
            actions::action_type action_;
            actions::continuation_type continuation_;
        };

        /// support functions for boost::intrusive_ptr
        inline void intrusive_ptr_add_ref(parcel_data* p)
        {
            ++p->count_;
        }

        inline void intrusive_ptr_release(parcel_data* p)
        {
            if (0 == --p->count_)
                delete p;
        }

        ///////////////////////////////////////////////////////////////////////
        // part of the parcel data is stored in a contigious memory area, which
        // helps improving serialization times

        // single-destination buffer
        class single_destination_parcel_data : public parcel_data
        {
        public:
            single_destination_parcel_data()
            {
                data_.start_time_ = 0;
                data_.creation_time_ = 0;
                data_.has_source_id_ = 0;
                data_.has_continuation_ = 0;
            }

            single_destination_parcel_data(naming::id_type const& apply_to,
                    naming::address const& addr, actions::base_action* act)
              : parcel_data(act)
            {
                data_.start_time_ = 0;
                data_.creation_time_ = 0;
                data_.has_source_id_ = 0;
                data_.has_continuation_ = 0;

                dest_ = apply_to;
                addr_ = addr;

                HPX_ASSERT(components::types_are_compatible(
                    act->get_component_type(), addr.type_));
            }

            single_destination_parcel_data(naming::id_type const& apply_to,
                    naming::address const& addr, actions::base_action* act,
                   actions::continuation* do_after)
              : parcel_data(act, do_after)
            {
                data_.start_time_ = 0;
                data_.creation_time_ = 0;
                data_.has_source_id_ = 0;
                data_.has_continuation_ = do_after ? 1 : 0;

                dest_ = apply_to;
                addr_ = addr;

                HPX_ASSERT(components::types_are_compatible(
                    act->get_component_type(), addr.type_));
            }

            single_destination_parcel_data(naming::id_type const& apply_to,
                    naming::address const& addr, actions::base_action* act,
                    actions::continuation_type do_after)
              : parcel_data(act, do_after)
            {
                data_.start_time_ = 0;
                data_.creation_time_ = 0;
                data_.has_source_id_ = 0;
                data_.has_continuation_ = do_after ? 1 : 0;

                dest_ = apply_to;
                addr_ = addr;

                HPX_ASSERT(components::types_are_compatible(
                    act->get_component_type(), addr.type_));
            }

            bool is_multi_destination() const { return false; }
            std::size_t size() const { return 1ul; }

            ///
            void set_start_time(double starttime)
            {
                data_.start_time_ = starttime;
                if (std::abs(data_.creation_time_) < 1e-10)
                    data_.creation_time_ = starttime;
            }
            double get_start_time() const
            {
                return data_.start_time_;
            }
            double get_creation_time() const
            {
                return data_.creation_time_;
            }

            /// get and set the destination id
            naming::id_type* get_destinations()
            {
                return &dest_;
            }
            naming::id_type const* get_destinations() const
            {
                return &dest_;
            }
            void set_destination(naming::id_type const& dest)
            {
                dest_ = dest;
            }
#if defined(HPX_SUPPORT_MULTIPLE_PARCEL_DESTINATIONS)
            void set_destinations(std::vector<naming::id_type> const& dests)
            {
                HPX_ASSERT(false);
            }
#endif
            /// get and set the destination address
            naming::address* get_destination_addrs()
            {
                return &addr_;
            }
            naming::address const* get_destination_addrs() const
            {
                return &addr_;
            }
            void set_destination_addr(naming::address const& addr)
            {
                addr_ = addr;
            }
#if defined(HPX_SUPPORT_MULTIPLE_PARCEL_DESTINATIONS)
            void set_destination_addrs(std::vector<naming::address> const& addrs)
            {
                HPX_ASSERT(false);
            }
#endif
            ///
            naming::locality const& get_destination_locality() const
            {
                return addr_.locality_;
            }

            naming::gid_type get_parcel_id() const
            {
                return data_.parcel_id_;
            }
            void set_parcel_id(naming::gid_type const& id)
            {
                data_.parcel_id_ = id;
            }

            std::size_t get_type_size() const
            {
                return sizeof(parcel_buffer) + this->get_action()->get_type_size();
            }

            template <typename Archive>
            void save(Archive& ar) const;

            template <typename Archive>
            void load(Archive& ar);

        private:
            friend std::ostream& operator<< (std::ostream& os,
                single_destination_parcel_data const& req);

            // serialization support
            template <typename Archive>
            void save_optimized(Archive& ar) const;
            template <typename Archive>
            void save_normal(Archive& ar) const;

            template <typename Archive>
            void load_optimized(Archive& ar);
            template <typename Archive>
            void load_normal(Archive& ar);

        private:
            // the parcel data is wrapped into a separate struct to simplify
            // serialization
            struct parcel_buffer
            {
                // actual parcel data
                naming::gid_type parcel_id_;
                double start_time_;
                double creation_time_;

                // data needed just for serialization purposes
                boost::uint64_t dest_size_;
                mutable boost::uint8_t has_source_id_;
                boost::uint8_t has_continuation_;
            };

            parcel_buffer data_;

            // other parcel data
            naming::id_type dest_;
            naming::address addr_;
        };

#if defined(HPX_SUPPORT_MULTIPLE_PARCEL_DESTINATIONS)
        ///////////////////////////////////////////////////////////////////////
        // multi-destination parcel buffer
        class multi_destination_parcel_data : public parcel_data
        {
        public:
            multi_destination_parcel_data()
            {
                data_.start_time_ = 0;
                data_.creation_time_ = 0;
                data_.dest_size_ = 0;
                data_.has_source_id_ = 0;
                data_.has_continuation_ = 0;
            }

            multi_destination_parcel_data(
                    std::vector<naming::id_type> const& apply_to,
                    std::vector<naming::address> const& addrs,
                    actions::action_type act)
              : parcel_data(act)
            {
                data_.start_time_ = 0;
                data_.creation_time_ = 0;
                data_.dest_size_ = apply_to.size();
                data_.has_source_id_ = 0;
                data_.has_continuation_ = 0;
                dests_ = apply_to;
                addrs_ = addrs;

#if defined(HPX_DEBUG)
                HPX_ASSERT(dests_.size() == addrs_.size());
                if (!dests_.empty() && addrs[0].locality_)
                {
                    // all destinations have to be on the same locality
                    naming::locality dest = get_destination_locality();
                    for (std::size_t i = 1; i != addrs.size(); ++i)
                    {
                        HPX_ASSERT(dest == addrs[i].locality_);
                    }

                    // all destination component types are properly matched
                    int comptype = act->get_component_type();
                    for (std::size_t i = 0; i != addrs.size(); ++i)
                    {
                        HPX_ASSERT(components::types_are_compatible(
                            comptype, addrs[i].type_));
                    }
                    HPX_UNUSED(comptype);
                }
#endif
            }

            bool is_multi_destination() const { return true; }
            std::size_t size() const { return data_.dest_size_; }

            ///
            void set_start_time(double starttime)
            {
                data_.start_time_ = starttime;
                if (std::abs(data_.creation_time_) < 1e-10)
                    data_.creation_time_ = starttime;
            }
            double get_start_time() const
            {
                return data_.start_time_;
            }
            double get_creation_time() const
            {
                return data_.creation_time_;
            }

            /// get and set the destination id
            naming::id_type* get_destinations()
            {
                return dests_.data();
            }
            naming::id_type const* get_destinations() const
            {
                return dests_.data();
            }
            void set_destination(naming::id_type const& dest)
            {
                HPX_ASSERT(false);
            }
            void set_destinations(std::vector<naming::id_type> const& dests)
            {
                dests_ = dests;
            }

            /// get and set the destination address
            naming::address* get_destination_addrs()
            {
                return addrs_.data();
            }
            naming::address const* get_destination_addrs() const
            {
                return addrs_.data();
            }
            void set_destination_addr(naming::address const& addr)
            {
                HPX_ASSERT(false);
            }
#if defined(HPX_SUPPORT_MULTIPLE_PARCEL_DESTINATIONS)
            void set_destination_addrs(std::vector<naming::address> const& addrs)
            {
                addrs_ = addrs;
            }
#endif
            ///
            naming::locality const& get_destination_locality() const
            {
                HPX_ASSERT(!addrs_.empty());
                return addrs_[0].locality_;
            }

            naming::gid_type get_parcel_id() const
            {
                return data_.parcel_id_;
            }
            void set_parcel_id(naming::gid_type const& id)
            {
                data_.parcel_id_ = id;
            }

            std::size_t get_type_size() const
            {
                return sizeof(parcel_buffer) +
                    traits::type_size<std::vector<naming::id_type> >::call(dests_) +
                    traits::type_size<std::vector<naming::address> >::call(addrs_) +
                    this->get_action()->get_type_size();      // action
            }

            template <typename Archive>
            void save(Archive& ar) const;

            template <typename Archive>
            void load(Archive& ar);

        private:
            friend std::ostream& operator<< (std::ostream& os,
                multi_destination_parcel_data const& req);

            // serialization support
            template <typename Archive>
            void save_optimized(Archive& ar) const;
            template <typename Archive>
            void save_normal(Archive& ar) const;

            template <typename Archive>
            void load_optimized(Archive& ar);
            template <typename Archive>
            void load_normal(Archive& ar);

        private:
            // the parcel data is wrapped into a separate struct to simplify
            // serialization
            struct parcel_buffer
            {
                // actual parcel data
                naming::gid_type parcel_id_;
                double start_time_;
                double creation_time_;

                // data needed just for serialization purposes
                boost::uint64_t dest_size_;
                mutable boost::uint8_t has_source_id_;
                boost::uint8_t has_continuation_;
            };
            parcel_buffer data_;

            // more parcel data
            std::vector<naming::id_type> dests_;
            std::vector<naming::address> addrs_;
        };
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    class HPX_EXPORT parcel
    {
    public:
        parcel() {}

        parcel(naming::id_type const& apply_to,
                naming::address const& addrs, actions::base_action* act)
          : data_(new detail::single_destination_parcel_data(apply_to, addrs, act))
        {}

#if defined(HPX_SUPPORT_MULTIPLE_PARCEL_DESTINATIONS)
        parcel(std::vector<naming::id_type> const& apply_to,
                std::vector<naming::address> const& addrs,
                actions::action_type act)
          : data_(new detail::multi_destination_parcel_data(apply_to, addrs, act))
        {}
#endif

        parcel(naming::id_type const& apply_to,
                naming::address const& addrs, actions::base_action* act,
                actions::continuation* do_after)
          : data_(new detail::single_destination_parcel_data(apply_to, addrs, act, do_after))
        {}

        parcel(naming::id_type const& apply_to,
                naming::address const& addrs, actions::base_action* act,
                actions::continuation_type do_after)
          : data_(new detail::single_destination_parcel_data(apply_to, addrs, act, do_after))
        {}

        ~parcel() {}

        // default copy constructor is ok
        // default assignment operator is ok

        actions::action_type get_action() const
        {
            return data_->get_action();
        }

        actions::continuation_type get_continuation() const
        {
            return data_->get_continuation();
        }

        /// get and set the source locality/component id
        naming::id_type& get_source()
        {
            return data_->get_source();
        }
        naming::id_type const& get_source() const
        {
            return data_->get_source();
        }

        void set_source(naming::id_type const& source_id)
        {
            data_->set_source(source_id);
        }

        std::size_t size() const
        {
            return data_->size();
        }

        /// get and set the destination id
        naming::id_type* get_destinations()
        {
            return data_->get_destinations();
        }
        naming::id_type const* get_destinations() const
        {
            return data_->get_destinations();
        }
#if defined(HPX_SUPPORT_MULTIPLE_PARCEL_DESTINATIONS)
        void set_destinations(std::vector<naming::id_type> const& dests)
        {
            if (data_->is_multi_destination()) {
                data_->set_destinations(dests);
            }
            else {
                HPX_ASSERT(dests.size() == 1);
                data_->set_destination(dests[0]);
            }
        }
#endif
        naming::locality const& get_destination_locality() const
        {
            return data_->get_destination_locality();
        }

        /// get and set the destination address
        naming::address* get_destination_addrs()
        {
            return data_->get_destination_addrs();
        }
        naming::address const* get_destination_addrs() const
        {
            return data_->get_destination_addrs();
        }
#if defined(HPX_SUPPORT_MULTIPLE_PARCEL_DESTINATIONS)
        void set_destination_addrs(std::vector<naming::address> const& addrs)
        {
            if (data_->is_multi_destination()) {
                data_->set_destination_addrs(addrs);
            }
            else {
                HPX_ASSERT(addrs.size() == 1);
                data_->set_destination_addr(addrs[0]);
            }
        }
#endif
        void set_start_time(double starttime)
        {
            data_->set_start_time(starttime);
        }
        double get_start_time() const
        {
            return data_->get_start_time();
        }
        double get_creation_time() const
        {
            return data_->get_creation_time();
        }

        threads::thread_priority get_thread_priority() const
        {
            return data_->get_thread_priority();
        }

        naming::gid_type get_parcel_id() const
        {
            return data_->get_parcel_id();
        }
        void set_parcel_id(naming::gid_type const& id)
        {
            data_->set_parcel_id(id);
        }

        util::binary_filter* get_serialization_filter() const
        {
            return data_->get_serialization_filter(*this);
        }

        policies::message_handler* get_message_handler(
            parcelset::parcelhandler* ph, naming::locality const& loc,
            parcelset::connection_type t) const
        {
            return data_->get_message_handler(ph, loc, t, *this);
        }

        std::size_t get_type_size() const
        {
            return data_->get_type_size();
        }

        // generate unique parcel id
        static naming::gid_type generate_unique_id(
            boost::uint32_t locality_id = naming::invalid_locality_id);

    private:
        friend std::ostream& operator<< (std::ostream& os, parcel const& req);

        // serialization support
        friend class boost::serialization::access;

        template <typename Archive>
        void save(Archive& ar, const unsigned int version) const;

        template <typename Archive>
        void load(Archive& ar, const unsigned int version);

        BOOST_SERIALIZATION_SPLIT_MEMBER()

    private:
        boost::intrusive_ptr<detail::parcel_data> data_;
    };
}}

namespace hpx { namespace traits
{
    template<>
    struct type_size<hpx::parcelset::parcel>
    {
        static std::size_t call(hpx::parcelset::parcel const& parcel_)
        {
            return sizeof(hpx::parcelset::parcel) + parcel_.get_type_size();
        }
    };
}}

namespace boost { namespace serialization
{
    template <>
    struct is_bitwise_serializable<
            hpx::parcelset::detail::single_destination_parcel_data::parcel_buffer>
       : boost::mpl::true_
    {};

#if defined(HPX_SUPPORT_MULTIPLE_PARCEL_DESTINATIONS)
    template <>
    struct is_bitwise_serializable<
            hpx::parcelset::detail::multi_destination_parcel_data::parcel_buffer>
       : boost::mpl::true_
    {};
#endif
}}

///////////////////////////////////////////////////////////////////////////////
// this is the current version of the parcel serialization format
// this definition needs to be in the global namespace
#if defined(__GNUG__) && !defined(__INTEL_COMPILER)
#if defined(HPX_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#pragma GCC diagnostic push
#endif
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif

BOOST_CLASS_TRACKING(hpx::parcelset::parcel, boost::serialization::track_never)
BOOST_CLASS_VERSION(hpx::parcelset::parcel, HPX_PARCEL_VERSION)

BOOST_CLASS_TRACKING(hpx::parcelset::detail::single_destination_parcel_data,
    boost::serialization::track_never)
#if defined(HPX_SUPPORT_MULTIPLE_PARCEL_DESTINATIONS)
BOOST_CLASS_TRACKING(hpx::parcelset::detail::multi_destination_parcel_data,
    boost::serialization::track_never)
#endif

#if defined(__GNUG__) && !defined(__INTEL_COMPILER)
#if defined(HPX_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#pragma GCC diagnostic pop
#endif
#endif

#include <hpx/traits/type_size.hpp>

#include <hpx/config/warnings_suffix.hpp>

#endif
