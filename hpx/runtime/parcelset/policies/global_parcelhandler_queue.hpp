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

#include <boost/assert.hpp>
#include <boost/lockfree/fifo.hpp>

namespace hpx { namespace parcelset { namespace policies
{
    ///////////////////////////////////////////////////////////////////////////
    struct global_parcelhandler_queue : parcelhandler_queue_base
    {
        global_parcelhandler_queue() :
            ph_(0), queue_length(0)
        {}

        ~global_parcelhandler_queue()
        {
            parcel p;
            while (get_parcel(p)) {}
        }

        void add_parcel(parcel const& p)
        {
            parcel* tmp = new parcel(p);

            // Add parcel to queue and increment queue length.
            parcels_.enqueue(tmp);
            ++queue_length;

            naming::address addr(tmp->get_destination_addr());

            // do some work (notify event handlers)
            BOOST_ASSERT(ph_ != 0);
            notify_(*ph_, addr);
        }

        bool get_parcel(parcel& p)
        {
            parcel* tmp;

            // Remove parcel from queue and decrement queue length.
            if (parcels_.dequeue(tmp))
            {
                std::swap(p, *tmp);
                delete tmp;
                --queue_length;
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

        boost::int64_t get_queue_length() const
        {
            return queue_length.load();
        }

      private:
        boost::lockfree::fifo<parcel*> parcels_;

        parcelhandler* ph_;

        boost::atomic<boost::int64_t> queue_length;

        boost::signals2::signal_type<
            void(parcelhandler&, naming::address const&)
          , boost::signals2::keywords::mutex_type<lcos::local::mutex>
        >::type notify_;
    };
}}}

#endif // HPX_894FCD94_A2A4_413D_AD50_088A9178DE77

