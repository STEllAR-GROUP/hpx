//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_PARCEL_COALESCING)
#include <hpx/runtime/parcelset/parcelport.hpp>
#include <hpx/util/unlock_guard.hpp>
#include <hpx/util/high_resolution_clock.hpp>
#include <hpx/util/get_and_reset_value.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/bind.hpp>

#include <hpx/plugins/message_handler_factory.hpp>
#include <hpx/plugins/parcel/coalescing_message_handler.hpp>
#include <hpx/plugins/parcel/coalescing_counter_registry.hpp>

#include <boost/lexical_cast.hpp>

#include <mutex>

#include <string>

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
                   "interval = 100\n"
                   "allow_background_flush = 1";
        }
    };
}}

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_PLUGIN_MODULE_DYNAMIC();
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

        bool get_background_flush()
        {
            std::string value = hpx::get_config_entry(
                "hpx.plugins.coalescing_message_handler.allow_background_flush",
                "1");
            return !value.empty() && value[0] != '0';
        }
    }

    coalescing_message_handler::coalescing_message_handler(
            char const* action_name, parcelset::parcelport* pp, std::size_t num,
            std::size_t interval)
      : pp_(pp), buffer_(detail::get_num_messages(num)),
        timer_(
            util::bind(&coalescing_message_handler::timer_flush, this_()),
            util::bind(&coalescing_message_handler::flush, this_(),
                parcelset::policies::message_handler::flush_mode_timer, true),
            detail::get_interval(interval), std::string(action_name) + "_timer",
            true),
        stopped_(false),
        allow_background_flush_(detail::get_background_flush()),
        num_parcels_(0), reset_num_parcels_(0),
            reset_num_parcels_per_message_parcels_(0),
        num_messages_(0), reset_num_messages_(0),
            reset_num_parcels_per_message_messages_(0),
        started_at_(util::high_resolution_clock::now()),
        reset_time_num_parcels_(0)
    {
        // register performance counter functions
        using util::placeholders::_1;
        coalescing_counter_registry::instance().register_action(action_name,
            util::bind(&coalescing_message_handler::get_parcels_count, this, _1),
            util::bind(&coalescing_message_handler::get_messages_count, this, _1),
            util::bind(&coalescing_message_handler::
                get_parcels_per_message_count, this, _1),
            util::bind(&coalescing_message_handler::
                get_average_time_between_parcels, this, _1));
    }

    void coalescing_message_handler::put_parcel(
        parcelset::locality const& dest, parcelset::parcel p,
        write_handler_type f)
    {
        std::unique_lock<mutex_type> l(mtx_);
        ++num_parcels_;

        if (stopped_) {
            ++num_messages_;
            l.unlock();

            // this instance should not buffer parcels anymore
            pp_->put_parcel(dest, std::move(p), f);
            return;
        }

        detail::message_buffer::message_buffer_append_state s =
            buffer_.append(dest, std::move(p), std::move(f));

        switch(s) {
        case detail::message_buffer::first_message:
            l.unlock();
            timer_.start(false);        // start deadline timer to flush buffer
            break;

        case detail::message_buffer::normal:
            if (timer_.is_started())
                break;

            l.unlock();
            timer_.start(false);        // start deadline timer to flush buffer
            break;

        case detail::message_buffer::buffer_now_full:
            flush_locked(l,
                parcelset::policies::message_handler::flush_mode_buffer_full,
                false);
            break;

        default:
            l.unlock();
            HPX_THROW_EXCEPTION(bad_parameter,
                "coalescing_message_handler::put_parcel",
                "unexpected return value from message_buffer::append");
            return;
        }
    }

    bool coalescing_message_handler::timer_flush()
    {
        // adjust timer if needed
        std::unique_lock<mutex_type> l(mtx_);
        if (!buffer_.empty())
        {
            flush_locked(l,
                parcelset::policies::message_handler::flush_mode_timer,
                false);
        }

        // do not restart timer for now, will be restarted on next parcel
        return false;
    }

    bool coalescing_message_handler::flush(
        parcelset::policies::message_handler::flush_mode mode,
        bool stop_buffering)
    {
        std::unique_lock<mutex_type> l(mtx_);
        return flush_locked(l, mode, stop_buffering);
    }

    bool coalescing_message_handler::flush_locked(
        std::unique_lock<mutex_type>& l,
        parcelset::policies::message_handler::flush_mode mode,
        bool stop_buffering)
    {
        HPX_ASSERT(l.owns_lock());

        // proceed with background work only if explicitly allowed
        if (!allow_background_flush_ &&
            mode == parcelset::policies::message_handler::flush_mode_background_work)
        {
            return false;
        }

        if (!stopped_ && stop_buffering) {
            stopped_ = true;

            util::unlock_guard<std::unique_lock<mutex_type> > ul(l);
            timer_.stop();              // interrupt timer
        }

        if (buffer_.empty())
            return false;

        detail::message_buffer buff (buffer_.capacity());
        std::swap(buff, buffer_);

        ++num_messages_;
        l.unlock();

        HPX_ASSERT(NULL != pp_);
        buff(pp_);                   // 'invoke' the buffer

        return true;
    }

    // performance counter values
    boost::int64_t
    coalescing_message_handler::get_average_time_between_parcels(bool reset)
    {
        std::unique_lock<mutex_type> l(mtx_);
        boost::int64_t now = util::high_resolution_clock::now();
        if (num_parcels_ == 0)
        {
            if (reset) started_at_ = now;
            return 0;
        }

        boost::int64_t num_parcels = num_parcels_ - reset_time_num_parcels_;
        if (num_parcels == 0)
        {
            if (reset) started_at_ = now;
            return 0;
        }

        HPX_ASSERT(now >= started_at_);
        boost::int64_t value = (now - started_at_) / num_parcels;

        if (reset)
        {
            started_at_ = now;
            reset_time_num_parcels_ = num_parcels_;
        }

        return value;
    }

    boost::int64_t coalescing_message_handler::get_parcels_count(bool reset)
    {
        std::unique_lock<mutex_type> l(mtx_);
        boost::int64_t num_parcels = num_parcels_ - reset_num_parcels_;
        if (reset)
            reset_num_parcels_ = num_parcels_;
        return num_parcels;
    }

    boost::int64_t
        coalescing_message_handler::get_parcels_per_message_count(bool reset)
    {
        std::unique_lock<mutex_type> l(mtx_);

        if (num_messages_ == 0)
        {
            if (reset)
            {
                reset_num_parcels_per_message_parcels_ = num_parcels_;
                reset_num_parcels_per_message_messages_ = num_messages_;
            }
            return 0;
        }

        boost::int64_t num_parcels =
            num_parcels_ - reset_num_parcels_per_message_parcels_;
        boost::int64_t
            num_messages = num_messages_ - reset_num_parcels_per_message_messages_;

        if (reset)
        {
            reset_num_parcels_per_message_parcels_ = num_parcels_;
            reset_num_parcels_per_message_messages_ = num_messages_;
        }

        if (num_messages == 0)
            return 0;

        return num_parcels / num_messages;
    }

    boost::int64_t coalescing_message_handler::get_messages_count(bool reset)
    {
        std::unique_lock<mutex_type> l(mtx_);
        boost::int64_t num_messages = num_messages_ - reset_num_messages_;
        if (reset)
            reset_num_messages_ = num_messages_;
        return num_messages;
    }

    ///////////////////////////////////////////////////////////////////////////
    // register the given action (called during startup)
    void coalescing_message_handler::register_action(char const* action,
        error_code& ec)
    {
        coalescing_counter_registry::instance().register_action(action);
        if (&ec != &throws)
            ec = make_success_code();
    }
}}}

#endif
