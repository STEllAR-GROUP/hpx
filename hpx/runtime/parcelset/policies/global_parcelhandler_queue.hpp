////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011 Katelyn Kufahl & Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_894FCD94_A2A4_413D_AD50_088A9178DE77)
#define HPX_894FCD94_A2A4_413D_AD50_088A9178DE77

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/parcelset/parcel.hpp>
#include <hpx/runtime/parcelset/parcelhandler.hpp>
#include <hpx/runtime/parcelset/parcelhandler_queue_base.hpp>
#include <hpx/lcos/local/spinlock.hpp>

#include <boost/assert.hpp>

#include <queue>

namespace hpx { namespace parcelset { namespace policies
{
    ///////////////////////////////////////////////////////////////////////////
    struct global_parcelhandler_queue : parcelhandler_queue_base
    {
    private:
        typedef lcos::local::spinlock mutex_type;
        typedef std::map<naming::gid_type, parcel> parcel_map_type;

    public:
        global_parcelhandler_queue()
          : ph_(0)
        {}

        ~global_parcelhandler_queue() {}

        bool add_parcel(parcel const& p)
        {
            naming::gid_type id(p.get_parcel_id());

            // Add parcel to queue.
            {
                mutex_type::scoped_lock l(mtx_);
                std::pair<parcel_map_type::iterator, bool> ret =
                    parcels_.insert(parcel_map_type::value_type(id, p));

                if (!ret.second) {
                    HPX_THROW_EXCEPTION(bad_parameter,
                        "global_parcelhandler_queue::add_parcel",
                        "Could not add received parcel to the parcelhandler "
                        "queue");
                    return false;
                }
            }

            // do some work (notify event handlers)
            BOOST_ASSERT(ph_ != 0);
            notify_(*ph_, id);
            return true;
        }

        bool add_exception(boost::exception_ptr e)
        {
            {
                mutex_type::scoped_lock l(mtx_);
                errors_.push(e);
            }

            // do some work (notify event handlers)
            BOOST_ASSERT(ph_ != 0);
            notify_(*ph_, naming::invalid_gid);
            return true;
        }

        bool get_parcel(parcel& p)
        {
            // Remove the first parcel from queue.
            mutex_type::scoped_lock l(mtx_);

            if (!errors_.empty()) {
                // handle pending exceptions first
                boost::exception_ptr e = errors_.front();
                errors_.pop();

                // now rethrow the topmost exception
                boost::rethrow_exception(e);
            }
            else if (!parcels_.empty()) {
                parcel_map_type::iterator front = parcels_.begin();
                p = (*front).second;
                parcels_.erase(front);
                return true;
            }

            return false;
        }

        bool get_parcel(parcel& p, naming::gid_type const& parcel_id)
        {
            // Remove the requested parcel from queue.
            mutex_type::scoped_lock l(mtx_);

            parcel_map_type::iterator it = parcels_.find(parcel_id);
            if (it != parcels_.end()) {
                p = (*it).second;
                parcels_.erase(it);
                return true;
            }

            return false;
        }

        bool register_event_handler(callback_type const& sink)
        {
            return notify_.connect(sink).connected();
        }

        bool register_event_handler(callback_type const& sink
          , connection_type& conn)
        {
            return (conn = notify_.connect(sink)).connected();
        }

        void set_parcelhandler(parcelhandler* ph)
        {
            BOOST_ASSERT(ph_ == 0);
            ph_ = ph;
        }

        std::size_t get_queue_length() const
        {
            mutex_type::scoped_lock l(mtx_);
            return parcels_.size();
        }

    private:
        mutable mutex_type mtx_;
        parcel_map_type parcels_;
        std::queue<boost::exception_ptr> errors_;

        parcelhandler* ph_;

        boost::signals2::signal_type<
            void(parcelhandler&, naming::gid_type)
          , boost::signals2::keywords::mutex_type<lcos::local::mutex>
        >::type notify_;
    };
}}}

#endif // HPX_894FCD94_A2A4_413D_AD50_088A9178DE77

