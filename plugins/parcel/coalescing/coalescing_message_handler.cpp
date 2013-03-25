//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/parcelset/parcelport.hpp>

#include <hpx/plugins/message_handler_factory.hpp>
#include <hpx/plugins/parcel/coalescing_message_handler.hpp>

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_PLUGIN_MODULE();
HPX_REGISTER_MESSAGE_HANDLER_FACTORY(
    hpx::plugins::parcel::coalescing_message_handler,
    coalescing_message_handler);

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
            timer_.restart();           // restart timer
            break;

        case detail::message_buffer::buffer_now_full:
            timer_.stop();              // interrupt timer

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
        if (!buffer_.empty()) {
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
