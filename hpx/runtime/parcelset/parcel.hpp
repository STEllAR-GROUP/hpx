//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2007 Richard D Guidry Jr
//  Copyright (c) 2007 Alexandre (aka Alex) TABBAL
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_PARCEL_MAR_26_2008_1051AM)
#define HPX_PARCELSET_PARCEL_MAR_26_2008_1051AM

#include <boost/serialization/split_member.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/version.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/naming/address.hpp>

#include <boost/intrusive_ptr.hpp>
#include <boost/assert.hpp>
#include <boost/detail/atomic_count.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
//  parcel serialization format version
#define HPX_PARCEL_VERSION 0x70

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////////
        class HPX_EXPORT parcel_data
        {
        public:
            parcel_data()
              : count_(0), start_time_(0), creation_time_(0)
            {}

            parcel_data(naming::gid_type const& apply_to,
                    naming::address const& addrs, actions::base_action* act)
              : count_(0), gids_(1), addrs_(1), action_(act),
                start_time_(0), creation_time_(0)
            {
                gids_[0] = apply_to;
                addrs_[0] = addrs;
            }

            parcel_data(std::vector<naming::gid_type> const& apply_to,
                    std::vector<naming::address> const& addrs,
                    actions::action_type act)
              : count_(0), gids_(apply_to), addrs_(addrs), action_(act),
                start_time_(0), creation_time_(0)
            {
#if defined(HPX_DEBUG)
                BOOST_ASSERT(gids_.size() == addrs_.size());
                if (!gids_.empty())
                {
                    // all destinations have to be on the same locality
                    naming::locality dest = get_destination_locality();
                    for (std::size_t i = 1; i < addrs.size(); ++i)
                    {
                        BOOST_ASSERT(dest == addrs[i].locality_);
                    }
                }
#endif
            }

            parcel_data(naming::gid_type const& apply_to,
                    naming::address const& addrs, actions::base_action* act,
                   actions::continuation* do_after)
              : count_(0), gids_(1), addrs_(1),
                action_(act), continuation_(do_after),
                start_time_(0), creation_time_(0)
            {
                gids_[0] = apply_to;
                addrs_[0] = addrs;
            }

            parcel_data(naming::gid_type const& apply_to,
                    naming::address const& addrs, actions::base_action* act,
                    actions::continuation_type do_after)
              : count_(0), gids_(1), addrs_(1),
                action_(act), continuation_(do_after),
                start_time_(0), creation_time_(0)
            {
                gids_[0] = apply_to;
                addrs_[0] = addrs;
            }

            ~parcel_data() {}

            // default copy constructor is ok
            // default assignment operator is ok

            actions::action_type get_action()
            {
                return action_;
            }

            actions::continuation_type get_continuation() const
            {
                return continuation_;
            }

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

            /// get and set the destination id
            std::vector<naming::gid_type>& get_destinations()
            {
                return gids_;
            }
            std::vector<naming::gid_type> const& get_destinations() const
            {
                return gids_;
            }
            void set_destinations(std::vector<naming::gid_type> const& dests)
            {
                gids_ = dests;
            }

            /// get and set the destination address
            void set_destination_addrs(std::vector<naming::address> const& addrs)
            {
                addrs_ = addrs;
            }
            std::vector<naming::address>& get_destination_addrs()
            {
                return addrs_;
            }
            std::vector<naming::address> const& get_destination_addrs() const
            {
                return addrs_;
            }

            naming::locality const& get_destination_locality() const
            {
                BOOST_ASSERT(!addrs_.empty());
                return addrs_[0].locality_;
            }

            void set_start_time(double starttime)
            {
                start_time_ = starttime;
                if (std::abs(creation_time_) < 1e-10)
                    creation_time_ = starttime;
            }
            double get_start_time() const
            {
                return start_time_;
            }
            double get_creation_time() const
            {
                return creation_time_;
            }

            threads::thread_priority get_thread_priority() const
            {
                BOOST_ASSERT(action_);
                return action_->get_thread_priority();
            }

            naming::gid_type get_parcel_id() const
            {
                return parcel_id_;
            }
            void set_parcel_id(naming::gid_type const& id)
            {
                parcel_id_ = id;
            }

        private:
            friend std::ostream& operator<< (std::ostream& os, parcel_data const& req);

            // serialization support
            friend class boost::serialization::access;

            template<class Archive>
            void save(Archive & ar, const unsigned int version) const;

            template<class Archive>
            void load(Archive & ar, const unsigned int version);

            BOOST_SERIALIZATION_SPLIT_MEMBER()

        private:
            friend void intrusive_ptr_add_ref(parcel_data* p);
            friend void intrusive_ptr_release(parcel_data* p);

            boost::detail::atomic_count count_;
            naming::gid_type parcel_id_;
            std::vector<naming::gid_type> gids_;
            std::vector<naming::address> addrs_;
            naming::id_type source_id_;
            actions::action_type action_;
            actions::continuation_type continuation_;
            double start_time_;
            double creation_time_;
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
    }

    ///////////////////////////////////////////////////////////////////////////
    class HPX_EXPORT parcel
    {
    public:
        parcel() {}

        parcel(naming::gid_type const& apply_to,
                naming::address const& addrs, actions::base_action* act)
          : data_(new detail::parcel_data(apply_to, addrs, act))
        {}

        parcel(std::vector<naming::gid_type> const& apply_to,
                std::vector<naming::address> const& addrs,
                actions::action_type act)
          : data_(new detail::parcel_data(apply_to, addrs, act))
        {}

        parcel(naming::gid_type const& apply_to,
                naming::address const& addrs, actions::base_action* act,
                actions::continuation* do_after)
          : data_(new detail::parcel_data(apply_to, addrs, act, do_after))
        {}

        parcel(naming::gid_type const& apply_to,
                naming::address const& addrs, actions::base_action* act,
                actions::continuation_type do_after)
          : data_(new detail::parcel_data(apply_to, addrs, act, do_after))
        {}

        ~parcel() {}

        // default copy constructor is ok
        // default assignment operator is ok

        actions::action_type get_action()
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

        naming::gid_type& get_source_gid()
        {
            return get_source().get_gid();
        }
        naming::gid_type const& get_source_gid() const
        {
            return get_source().get_gid();
        }

        void set_source(naming::id_type const& source_id)
        {
            data_->set_source(source_id);
        }

        /// get and set the destination id
        std::vector<naming::gid_type>& get_destinations()
        {
            return data_->get_destinations();
        }
        std::vector<naming::gid_type> const& get_destinations() const
        {
            return data_->get_destinations();
        }
        void set_destinations(std::vector<naming::gid_type> const& dests)
        {
            data_->set_destinations(dests);
        }

        naming::locality const& get_destination_locality() const
        {
            return data_->get_destination_locality();
        }

        /// get and set the destination address
        void set_destination_addrs(std::vector<naming::address> const& addrs)
        {
            data_->set_destination_addrs(addrs);
        }
        std::vector<naming::address>& get_destination_addrs()
        {
            return data_->get_destination_addrs();
        }
        std::vector<naming::address> const& get_destination_addrs() const
        {
            return data_->get_destination_addrs();
        }

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

        // generate unique parcel id
        static naming::gid_type generate_unique_id();

    private:
        friend std::ostream& operator<< (std::ostream& os, parcel const& req);

        // serialization support
        friend class boost::serialization::access;

        template <typename Archive>
        void serialize(Archive & ar, const unsigned int version);

    private:
        boost::intrusive_ptr<detail::parcel_data> data_;
    };
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

#if defined(__GNUG__) && !defined(__INTEL_COMPILER)
#if defined(HPX_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#pragma GCC diagnostic pop
#endif
#endif

#include <hpx/config/warnings_suffix.hpp>

#endif
