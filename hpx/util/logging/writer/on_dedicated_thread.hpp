// on_dedicated_thread.hpp

// Boost Logging library
//
// Author: John Torjo, www.torjo.com
//
// Copyright (C) 2007 John Torjo (see www.torjo.com for email)
//
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org for updates, documentation, and revision history.
// See http://www.torjo.com/log2/ for more details


#ifndef JT28092007_on_dedicated_thread_HPP_DEFINED
#define JT28092007_on_dedicated_thread_HPP_DEFINED

#if defined(HPX_MSVC) && (HPX_MSVC >= 1020)
# pragma once
#endif

#include <boost/version.hpp>
#include <hpx/util/logging/detail/fwd.hpp>
#include <hpx/util/logging/detail/forward_constructor.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/xtime.hpp>
#include <boost/bind.hpp>
#include <hpx/util/logging/detail/manipulator.hpp> // hpx::util::logging::manipulator

#include <vector>

namespace hpx { namespace util { namespace logging { namespace writer {

/** @file hpx/util/logging/writer/on_dedidcated_thread.hpp
*/

namespace detail {
    template<class msg_type> struct dedicated_thread_context {
        dedicated_thread_context() : is_working(true), write_period_ms(100),
            is_paused(false), pause_acknowledged (false) {}

        bool is_working;
        int write_period_ms;

        // if true, we've paused writing
        // this is useful for when you want to manipulate the formatters/destinations
        // (for instance, when you want to delete a destination)
        bool is_paused;
        // when true, the dedicated thread has acknowledged the pause
        bool pause_acknowledged ;

        hpx::util::logging::threading::mutex cs;

        // the thread doing the write
        typedef boost::shared_ptr<boost::thread> thread_ptr;
        thread_ptr writer;

        // ... so that reallocations are fast
        typedef boost::shared_ptr<msg_type> ptr;
        typedef std::vector<ptr> array;
        array msgs;
    };
}

/**
@brief Performs all writes on a dedicated thread
- very efficient and <b>thread-safe</b>.

<tt>\#include <hpx/util/logging/writer/on_dedicated_thread.hpp> </tt>

Keeps locks in the worker threads to a minimum:
whenever a message is logged, is put into a queue
(this is how long the lock lasts).
Then, a dedicated thread reads the queue, and processes the messages
(applying formatters and destinations if needed).

@section on_dedicated_thread_logger Transforming a logger into
on-dedicated-thread writer

To transform a @b logger into on-dedicated-thread (thread-safe) writer,
simply specify @c on_dedicated_thread as the thread safety:

@code
typedef logger_format_write< default_, default_,
writer::threading::on_dedicated_thread > logger_type;
@endcode

Of if you're using @ref hpx::util::logging::scenario::usage scenarios,
specify @c speed for the @c logger::favor_ :
@code
using namespace hpx::util::logging::scenario::usage;
typedef use< ..., ..., ..., logger_::favor::speed> finder;
@endcode



\n\n
@section on_dedicated_thread_writer Transforming a writer into on-dedicated-thread writer

To transform a @b writer into on-dedicated-thread thread-safe writer,
simply surround the writer with @c on_dedicated_thread:

Example:

@code
typedef gather::ostream_like::return_str<> string;

// not thread-safe
logger< string, write_to_cout> g_l();

// thread-safe, on dedicated thread
logger< string, on_dedicated_thread<string,write_to_cout> > g_l();
@endcode

You should note that a @b writer is not necessary a %logger. It can be a destination,
for instance. For example, you might have a destination
where writing is time consuming, while writing to the rest of the destinations
is very fast.
You can choose to write to all but that destination on the current thread,
and to that destination on a dedicated thread.
(If you want to write to all destinations on a different thread,
we can go back to @ref on_dedicated_thread_logger "transforming a logger...")

*/
template<class msg_type, class base_type>
struct on_dedicated_thread
        : base_type,
          hpx::util::logging::manipulator::non_const_context<detail
    ::dedicated_thread_context<msg_type> > {

    typedef on_dedicated_thread<msg_type,base_type> self_type;
    typedef typename detail::dedicated_thread_context<msg_type> context_type;
    typedef typename hpx::util::logging::manipulator
        ::non_const_context<detail::dedicated_thread_context<msg_type> >
        non_const_context_base;

    typedef hpx::util::logging::threading::mutex::scoped_lock scoped_lock;

    HPX_LOGGING_FORWARD_CONSTRUCTOR(on_dedicated_thread,base_type)

    /**
        @brief Sets the write period : on the dedicated thread (in milliseconds)
    */
    void write_period_ms(int period_ms) {
        scoped_lock lk( non_const_context_base::context().cs);
        non_const_context_base::context().write_period_ms = period_ms;
    }

    ~on_dedicated_thread() {
        boost::shared_ptr<boost::thread> writer;
        { scoped_lock lk( non_const_context_base::context().cs);
          non_const_context_base::context().is_working = false;
          writer = non_const_context_base::context().writer;
        }

        if ( writer)
            writer->join();

        // write last messages, if any
        write_array();
    }

    void operator()(msg_type & msg) const {
        typedef typename context_type::ptr ptr;
        typedef typename context_type::thread_ptr thread_ptr;
        //ptr new_msg(new msg_type(msg));
        ptr new_msg(new msg_type);
        std::swap(msg, *new_msg);

        scoped_lock lk( non_const_context_base::context().cs);
        if ( !non_const_context_base::context().writer)
            non_const_context_base::context().writer = thread_ptr(
                new boost::thread( boost::bind(&self_type::do_write,this) ));

        non_const_context_base::context().msgs.push_back(new_msg);
    }

    /** @brief Resumes the writes, after a pause()
    */
    void resume() {
        scoped_lock lk( non_const_context_base::context().cs);
        non_const_context_base::context().is_paused = false;
    }

    /** @brief Pauses the writes, so that you can manipulate the base object
    (the formatters/destinations, for instance)

    After this function has been called, you can be @b sure that the other
    (dedicated) thread is not writing any messagges.
    In other words, the other thread is not manipulating the base object
    (formatters/destinations, for instance), but you can do it.

    FIXME allow a timeout as well
    */
    void pause() {
        { scoped_lock lk( non_const_context_base::context().cs);
          non_const_context_base::context().is_paused = true;
          non_const_context_base::context().pause_acknowledged = false;
        }

        while ( true) {
            do_sleep(10);
            scoped_lock lk( non_const_context_base::context().cs);
            if ( non_const_context_base::context().pause_acknowledged )
                // the other thread has acknowledged
                break;
        }
    }
private:
    static void do_sleep (int sleep_ms) {
        const int NANOSECONDS_PER_SECOND = 1000 * 1000 * 1000;
        boost::xtime to_wait;
#if BOOST_VERSION < 105000
        xtime_get(&to_wait, boost::TIME_UTC);
#else
        // V1.50 changes the name of boost::TIME_UTC
        xtime_get(&to_wait, boost::TIME_UTC_);
#endif
        to_wait.sec += sleep_ms / 1000;
        to_wait.nsec += (sleep_ms % 1000) * (NANOSECONDS_PER_SECOND / 1000);
        to_wait.sec += to_wait.nsec / NANOSECONDS_PER_SECOND ;
        to_wait.nsec %= NANOSECONDS_PER_SECOND ;
        boost::thread::sleep( to_wait);
    }

    // normally it sleeps for sleep_ms millisecs
    // however, if we've been paused, it stops waiting
    void wait_or_wake_up_on_pause(int sleep_ms) const {
        int PERIOD = 100;
        while ( sleep_ms > 0) {
            do_sleep(sleep_ms > PERIOD ? PERIOD : sleep_ms);
            sleep_ms -= PERIOD;

            scoped_lock lk( non_const_context_base::context().cs);
            if ( non_const_context_base::context().is_paused)
                // this way we wake up early after we've been pause()d,
                // even if sleep_ms has a high value
                break;
        }
    }

    void do_write() const {
        int sleep_ms = 0;
        while ( true) {
            { scoped_lock lk( non_const_context_base::context().cs);
              // refresh it - just in case it got changed...
              sleep_ms = non_const_context_base::context().write_period_ms;
              if ( !non_const_context_base::context().is_working)
                  break; // we've been destroyed
            }
            wait_or_wake_up_on_pause(sleep_ms);

            { scoped_lock lk( non_const_context_base::context().cs);
              if ( non_const_context_base::context().is_paused) {
                  // we're paused
                  non_const_context_base::context().pause_acknowledged = true;
                  continue;
              }
            }
            write_array();
            { scoped_lock lk( non_const_context_base::context().cs);
              if ( non_const_context_base::context().is_paused)
                // we're paused
                non_const_context_base::context().pause_acknowledged = true;
            }
        }
    }

protected:
    // note: this is virtual, so that if you want to do profiling,
    // you can (that is, you can override this as well
    virtual void write_array() const {
        typedef typename context_type::array array;

        array msgs;
        { scoped_lock lk( non_const_context_base::context().cs);
          std::swap( non_const_context_base::context().msgs, msgs);
          // reserve elements - so that we don't get automatically resized often
          non_const_context_base::context().msgs.reserve( msgs.size() );
        }

        for ( typename array::iterator b = msgs.begin(), e = msgs.end(); b != e; ++b)
            base_type::operator()(*(b->get()));
    }
};

}}}}

#endif

