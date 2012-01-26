//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/state.hpp>
#include <hpx/runtime.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/serialize_sequence.hpp>
#include <hpx/runtime/components/console_logging.hpp>
#include <hpx/runtime/components/server/console_logging.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/runtime/actions/continuation.hpp>

#include <boost/fusion/include/at_c.hpp>
#include <boost/assert.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    void fallback_console_logging_locked(messages_type const& msgs,
        std::string fail_msg = "")
    {
        using boost::fusion::at_c;

        if (!fail_msg.empty())
            fail_msg = "Logging failed due to: " + fail_msg + "\n";

        BOOST_FOREACH(message_type const& msg, msgs)
        {
            switch (at_c<0>(msg)) {
            default:
            case destination_hpx:
                LHPX_CONSOLE_(at_c<1>(msg)) << fail_msg << at_c<2>(msg);
                break;

            case destination_timing:
                LTIM_CONSOLE_(at_c<1>(msg)) << fail_msg << at_c<2>(msg);
                break;

            case destination_agas:
                LAGAS_CONSOLE_(at_c<1>(msg)) << fail_msg << at_c<2>(msg);
                break;

            case destination_app:
                LAPP_CONSOLE_(at_c<1>(msg)) << fail_msg << at_c<2>(msg);
                break;
            }
        }
    }

    void console_logging_locked(naming::id_type const& prefix,
        messages_type const& msgs)
    {
        // If we're not in an HPX thread, we cannot call apply as it may access
        // the AGAS components. We just throw an exception here - there's no
        // thread-manager, so the exception will probably be unhandled. This is
        // desirable in this situation, as we can't trust the logging system to
        // report this error.
        if (HPX_UNLIKELY(!threads::get_self_ptr())) {
            // write the message to a local file in any case
            fallback_console_logging_locked(msgs);

            // raise error as this should get called from outside a HPX-thread
            HPX_THROW_EXCEPTION(null_thread_id,
                "components::console_logging_locked",
                "console_logging_locked was not called from a HPX-thread");
        }

        try {
            applier::apply<server::console_logging_action<> >(prefix, msgs);
        }
        catch (hpx::exception const& e) {
            // if this is not the console locality (or any other error occurs)
            // we might be too late for any logging, write to local file
            fallback_console_logging_locked(msgs, e.what());
        }
    }

    void pending_logs::add(message_type const& msg)
    {
        {
            queue_mutex_type::scoped_lock l(queue_mtx_);
            queue_.push_back(msg);
            ++queue_size_;
        }

        if (0 == hpx::get_runtime_ptr()) {
            // the is_console call below would fail
            messages_type msgs;
            {
                queue_mutex_type::scoped_lock l(queue_mtx_);
                queue_.swap(msgs);
                queue_size_.store(0);
            }
            fallback_console_logging_locked(msgs);
        }
        else if (naming::get_agas_client().is_console()) {
            // invoke actual logging immediately if we're on the console
            send();
        }
        else if (max_pending < queue_size_.load()) {
            send();
        }
    }

    void pending_logs::cleanup()
    {
        send();
    }

    bool pending_logs::ensure_prefix()
    {
        // Resolve the console prefix if it's still invalid.
        if (HPX_UNLIKELY(naming::invalid_id == prefix_))
        {
            prefix_mutex_type::scoped_try_lock l(prefix_mtx_);

            if (l.owns_lock() && (naming::invalid_id == prefix_))
            {
                naming::gid_type raw_prefix;
                naming::get_agas_client().get_console_prefix(raw_prefix);
                BOOST_ASSERT(naming::invalid_gid != raw_prefix);
                prefix_ = naming::id_type(raw_prefix, naming::id_type::unmanaged);
            }

            // Someone else started getting the console prefix.
            else {
                return false;
            }
        }
        return true;
    }

    void pending_logs::send()
    {
        if (0 == queue_size_.load())
            return;         // nothing to send

        if (!ensure_prefix())
            return;         // some other thread tries to do logging

        messages_type msgs;
        {
            queue_mutex_type::scoped_lock l(queue_mtx_);
            queue_.swap(msgs);
            queue_size_.store(0);
        }
        BOOST_ASSERT(!msgs.empty());

        // we can only invoke send() from within a HPX-thread
        if (!(threads::threadmanager_is(running) && threads::get_self_ptr() &&
              activated_.load()))
        {
            // write locally into a file
            fallback_console_logging_locked(msgs);
            return;
        }

        console_logging_locked(prefix_, msgs);
    }

    void console_logging(logging_destination dest, std::size_t level,
        std::string const& s)
    {
        util::static_<pending_logs, pending_logs_tag> logs;
        message_type msg(dest, level, s);
        logs.get().add(msg);
    }

    void cleanup_logging()
    {
        util::static_<pending_logs, pending_logs_tag> logs;
        logs.get().cleanup();
    }

    void activate_logging()
    {
        util::static_<pending_logs, pending_logs_tag> logs;
        logs.get().activate();
    }
}}

