//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/parcelset/parcelport.hpp>

#include <hpx/plugins/parcel/coalescing_message_handler.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace plugins { namespace parcel
{
    coalescing_message_handler::coalescing_message_handler(
            char const* action_name, parcelset::parcelport* pp, std::size_t num,
            std::size_t interval)
      : buffer_(num), pp_(pp),
        timer_(boost::bind(&coalescing_message_handler::timer_flush, this_()),
            boost::bind(&coalescing_message_handler::flush, this_(), true),
            interval, action_name, true),
        interval_(interval),
        stopped_(false)
    {}

    coalescing_message_handler::~coalescing_message_handler()
    {}

    void coalescing_message_handler::put_parcel(parcelset::parcel& p,
        write_handler_type const& f)
    {
        if (stopped_) {
            // this instance should not buffer parcels anymore
            pp_->put_parcel(p, f);
            return;
        }

        detail::message_buffer::message_buffer_append_state s =
            buffer_.append(p, f);

        switch(s) {
        case detail::message_buffer::first_message:
            timer_.start(false);        // start deadline timer to flush buffer
            break;

        case detail::message_buffer::normal:
            break;

        case detail::message_buffer::buffer_now_full:
            BOOST_ASSERT(NULL != pp_);
            buffer_(pp_);               // 'invoke' the buffer
            break;

        default:
            HPX_THROW_EXCEPTION(bad_parameter,
                "coalescing_message_handler::put_parcel",
                "unexpected return value from message_buffer::append");
            return;
        }
    }

    bool coalescing_message_handler::timer_flush()
    {
        // adjust timer if needed
        if (buffer_.size()) {
            double ratio = buffer_.fill_ratio();
            if (ratio > 0.0 && ratio <= 0.1)
                timer_.slow_down(interval_*10);
            else if (ratio > 0.9)
                timer_.speed_up(interval_/10);

            flush();
            return true;        // restart timer
        }

        // do not restart timer for now, will be restarted on next parcel
        return false;
    }

    void coalescing_message_handler::flush(bool stop_buffering)
    {
        if (stop_buffering)
            stopped_ = true;

        BOOST_ASSERT(NULL != pp_);
        buffer_(pp_);                   // 'invoke' the buffer
    }
}}}
