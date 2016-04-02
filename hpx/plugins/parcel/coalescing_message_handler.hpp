//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_PARCELSET_COALESCING_MESSAGE_HANDLER_FEB_24_2013_0302PM)
#define HPX_RUNTIME_PARCELSET_COALESCING_MESSAGE_HANDLER_FEB_24_2013_0302PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_PARCEL_COALESCING)

#include <hpx/util/pool_timer.hpp>
#include <hpx/util/detail/count_num_args.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/runtime/parcelset/policies/message_handler.hpp>

#include <hpx/plugins/parcel/message_buffer.hpp>

#include <boost/preprocessor/stringize.hpp>
#include <boost/thread/locks.hpp>
#include <boost/cstdint.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace plugins { namespace parcel
{
    struct HPX_LIBRARY_EXPORT coalescing_message_handler
      : parcelset::policies::message_handler
    {
    private:
        coalescing_message_handler* this_() { return this; }

        typedef lcos::local::spinlock mutex_type;

    public:
        typedef parcelset::policies::message_handler::write_handler_type
            write_handler_type;

        coalescing_message_handler(char const* action_name,
            parcelset::parcelport* pp, std::size_t num = std::size_t(-1),
            std::size_t interval = std::size_t(-1));

        void put_parcel(parcelset::locality const & dest,
            parcelset::parcel p, write_handler_type f);

        bool flush(bool stop_buffering = false);

        // access performance counter data
        boost::int64_t get_parcels_count(bool reset);
        boost::int64_t get_messages_count(bool reset);
        boost::int64_t get_parcels_per_message_count(bool reset);
        boost::int64_t get_average_time_between_parcels(bool reset);

        // register the given action
        static void register_action(char const* action, error_code& ec);

    protected:
        bool timer_flush();
        bool flush_locked(boost::unique_lock<mutex_type>& l,
            bool stop_buffering);

    private:
        mutable mutex_type mtx_;
        parcelset::parcelport* pp_;
        detail::message_buffer buffer_;
        util::pool_timer timer_;
        bool stopped_;

        // performance counter data
        boost::int64_t num_parcels_;
        boost::int64_t reset_num_parcels_;
        boost::int64_t reset_num_parcels_per_message_parcels_;
        boost::int64_t num_messages_;
        boost::int64_t reset_num_messages_;
        boost::int64_t reset_num_parcels_per_message_messages_;
        boost::int64_t started_at_;
        boost::int64_t reset_time_num_parcels_;
    };
}}}

#include <hpx/config/warnings_suffix.hpp>

#endif
#endif
