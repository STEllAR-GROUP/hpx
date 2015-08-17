//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

#if defined(HPX_HAVE_PARCEL_COALESCING)
#include <hpx/runtime/parcelset/parcelport.hpp>
#include <hpx/util/unlock_guard.hpp>

#include <hpx/plugins/message_handler_factory.hpp>
#include <hpx/plugins/parcel/coalescing_message_handler.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/thread/locks.hpp>

namespace hpx { namespace traits
{
    // Inject additional configuration data into the factory registry for this
    // type. This information ends up in the system wide configuration database
    // under the plugin specific section:
    //
    //      [hpx.plugins.coalescing_message_handler]
    //      ...
    //      num_messages = 50
    //      interval = 100
    //
    template <>
    struct plugin_config_data<hpx::plugins::parcel::coalescing_message_handler>
    {
        static char const* call()
        {
            return "num_messages = 50\n"
                   "interval = 100";
        }
    };
}}

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_PLUGIN_MODULE();
HPX_REGISTER_MESSAGE_HANDLER_FACTORY(
    hpx::plugins::parcel::coalescing_message_handler,
    coalescing_message_handler);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace plugins { namespace parcel
{
    namespace detail
    {
        std::size_t get_num_messages(std::size_t num_messages)
        {
            if (std::size_t(-1) != num_messages)
                return num_messages;

            return boost::lexical_cast<std::size_t>(hpx::get_config_entry(
                "hpx.plugins.coalescing_message_handler.num_messages", 50));
        }

        std::size_t get_interval(std::size_t interval)
        {
            if (std::size_t(-1) != interval)
                return interval;

            return boost::lexical_cast<std::size_t>(hpx::get_config_entry(
                "hpx.plugins.coalescing_message_handler.interval", 100));
        }
    }

    coalescing_message_handler::coalescing_message_handler(
            char const* action_name, parcelset::parcelport* pp, std::size_t num,
            std::size_t interval)
      : pp_(pp), buffer_(detail::get_num_messages(num)),
        timer_(boost::bind(&coalescing_message_handler::timer_flush, this_()),
            boost::bind(&coalescing_message_handler::flush, this_(), true),
            detail::get_interval(interval), std::string(action_name) + "_timer",
            true),
        stopped_(false)
    {}

    void coalescing_message_handler::put_parcel(
        parcelset::locality const & dest, parcelset::parcel p,
        write_handler_type f)
    {
        boost::unique_lock<mutex_type> l(mtx_);
        if (stopped_) {
            l.unlock();

            // this instance should not buffer parcels anymore
            pp_->put_parcel(dest, std::move(p), f);
            return;
        }

        detail::message_buffer::message_buffer_append_state s =
            buffer_.append(dest, std::move(p), f);

        switch(s) {
        case detail::message_buffer::first_message:
            l.unlock();
            timer_.start(false);        // start deadline timer to flush buffer
            break;

        case detail::message_buffer::normal:
            l.unlock();
            timer_.restart(false);      // restart timer
            break;

        case detail::message_buffer::buffer_now_full:
            flush(l, false);
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
        boost::unique_lock<mutex_type> l(mtx_);
        if (!buffer_.empty())
            flush(l, false);

        // do not restart timer for now, will be restarted on next parcel
        return false;
    }

    bool coalescing_message_handler::flush(bool stop_buffering)
    {
        boost::unique_lock<mutex_type> l(mtx_);
        return flush(l, stop_buffering);
    }

    bool coalescing_message_handler::flush(
        boost::unique_lock<mutex_type>& l,
        bool stop_buffering)
    {
        HPX_ASSERT(l.owns_lock());

        if (!stopped_ && stop_buffering) {
            stopped_ = true;

            util::unlock_guard<boost::unique_lock<mutex_type> > ul(l);
            timer_.stop();              // interrupt timer
        }

        if (buffer_.empty())
            return false;

        detail::message_buffer buff (buffer_.capacity());
        std::swap(buff, buffer_);

        l.unlock();

        HPX_ASSERT(NULL != pp_);
        buff(pp_);                   // 'invoke' the buffer

        return true;
    }
}}}

#endif
