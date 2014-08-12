//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/state.hpp>
#include <hpx/runtime.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/serialize_sequence.hpp>
#include <hpx/runtime/components/console_logging.hpp>
#include <hpx/runtime/components/server/console_logging.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/util/reinitializable_static.hpp>

#include <boost/fusion/include/at_c.hpp>
#include <boost/serialization/vector.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    void fallback_console_logging_locked(messages_type const& msgs,
        std::string fail_msg = "")
    {
        using boost::fusion::at_c;

        if (!fail_msg.empty())
            fail_msg = "Logging failed due to: " + fail_msg + "\n";

        for (std::size_t i = 0; i != msgs.size(); ++i)
        {
            message_type const& msg = msgs[i];
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

            case destination_debuglog:
                LDEB_CONSOLE_ << fail_msg << at_c<2>(msg);
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
        if (HPX_UNLIKELY(!threads::get_self_ptr()))
        {
            // write the message to a local file in any case
            fallback_console_logging_locked(msgs);

            // raise error as this should get called from outside a HPX-thread
            HPX_THROW_EXCEPTION(null_thread_id,
                "components::console_logging_locked",
                "console_logging_locked was not called from a HPX-thread");
        }

        try {
            hpx::apply<server::console_logging_action<> >(prefix, msgs);
        }
        catch (hpx::exception const& e) {
            // if this is not the console locality (or any other error occurs)
            // we might be too late for any logging, write to local file
            fallback_console_logging_locked(msgs, e.what());
        }
    }

    bool pending_logs::is_active()
    {
        return threads::threadmanager_is(running) && threads::get_self_ptr() &&
            activated_.load();
    }

    void pending_logs::add(message_type const& msg)
    {
        if (0 == hpx::get_runtime_ptr()) {
            // This branch will be taken if it's too early or too late in the
            // game. We do local logging only. Any queued messages which may be
            // still left in the queue are logged locally as well.

            // queue up the new message and log it with the rest of it
            messages_type msgs;
            {
                queue_mutex_type::scoped_lock l(queue_mtx_);
                queue_.push_back(msg);
                queue_.swap(msgs);
            }

            fallback_console_logging_locked(msgs);
        }
        else if (is_active()) {
            // This branch will be taken under normal circumstances. We queue
            // up the message and either log it immediately (if we are on the
            // console locality) or defer logging until 'max_pending' messages
            // have been queued. Note: we can invoke send only from within a
            // HPX-thread.
            std::size_t size = 0;

            {
                queue_mutex_type::scoped_lock l(queue_mtx_);
                queue_.push_back(msg);
                size = queue_.size();
            }

            // Invoke actual logging immediately if we're on the console or
            // if the number of waiting log messages is too large.
            if (naming::get_agas_client().is_console() || size > max_pending)
                send();
        }
        else {
            // This branch will be taken if the runtime is up and running, but
            // either the thread manager is not active or this is not invoked
            // on a HPX-thread.

            // Note: is_console can be called outside of a HPX-thread
            if (!naming::get_agas_client().is_console())
            {
                // queue it for delivery to the console
                queue_mutex_type::scoped_lock l(queue_mtx_);
                queue_.push_back(msg);
            }
            else
            {
                // log it locally on the console
                messages_type msgs;
                msgs.push_back(msg);
                fallback_console_logging_locked(msgs);
            }
        }
    }

    void pending_logs::cleanup()
    {
        if (threads::threadmanager_is(running) && threads::get_self_ptr())
        {
            send();
        }
        else
        {
            messages_type msgs;
            {
                queue_mutex_type::scoped_lock l(queue_mtx_);
                if (queue_.empty())
                    return;         // some other thread did the deed
                queue_.swap(msgs);
            }

            fallback_console_logging_locked(msgs);
        }
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
                {
                    util::scoped_unlock<prefix_mutex_type::scoped_try_lock> ul(l);
                    naming::get_agas_client().get_console_locality(raw_prefix);
                }

                HPX_ASSERT(naming::invalid_gid != raw_prefix);
                if (!prefix_) {
                    prefix_ = naming::id_type(raw_prefix, naming::id_type::unmanaged);
                }
                else {
                    HPX_ASSERT(prefix_.get_gid() == raw_prefix);
                }
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
        // WARNING: Never, ever call this outside of a HPX-thread.
        HPX_ASSERT(threads::get_self_ptr());

        bool expected = false;
        if (!is_sending_.compare_exchange_strong(expected, true))
            return;

        try {
            {
                queue_mutex_type::scoped_lock l(queue_mtx_);
                if (queue_.empty())
                    return;         // some other thread did the deed
            }

            if (!ensure_prefix())
                return;             // some other thread tries to do logging

            messages_type msgs;
            {
                queue_mutex_type::scoped_lock l(queue_mtx_);
                if (queue_.empty())
                    return;         // some other thread did the deed
                queue_.swap(msgs);
            }

            console_logging_locked(prefix_, msgs);
        }
        catch(...) {
            is_sending_ = false;
            throw;
        }

        is_sending_ = false;
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        pending_logs& logger()
        {
            util::reinitializable_static<pending_logs, pending_logs_tag> logs;
            return logs.get();
        }
    }

    void console_logging(logging_destination dest, std::size_t level,
        std::string const& s)
    {
        message_type msg(dest, level, s);
        detail::logger().add(msg);
    }

    void cleanup_logging()
    {
        detail::logger().cleanup();
    }

    void activate_logging()
    {
        detail::logger().activate();
    }
}}

