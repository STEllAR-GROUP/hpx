////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <boost/ref.hpp>
#include <boost/bind.hpp>

#include <hpx/runtime.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/components/iostreams/server/output_stream.hpp>

#include <iostream>

namespace hpx { namespace iostreams { namespace server
{

///////////////////////////////////////////////////////////////////////////////
void output_stream::call_write_async(
    util::serializable_shared_ptr<std::deque<char> > const& in
) { // {{{
    mutex_type::scoped_lock l(mtx);

    // Perform the IO operation.
    write_f(*in);
} // }}}

void output_stream::write_async(
    util::serializable_shared_ptr<std::deque<char> > const& in
) { // {{{
    // Perform the IO in another OS thread. 
    get_runtime().get_io_pool().get_io_service().post(boost::bind
        (&output_stream::call_write_async, this, in));
} // }}}

///////////////////////////////////////////////////////////////////////////////
void output_stream::call_write_sync(
    util::serializable_shared_ptr<std::deque<char> > const& in
  , threads::thread_id_type caller
) {
    {
        mutex_type::scoped_lock l(mtx);

        // Perform the IO operation.
        write_f(*in);
    }

    // Wake up caller.
    threads::set_thread_state(caller, threads::pending);
}

void output_stream::write_sync(
    util::serializable_shared_ptr<std::deque<char> > const& in
) { // {{{
    threads::thread_self& self = threads::get_self();
    threads::thread_id_type id = self.get_thread_id();

    // Perform the IO in another OS thread. 
    get_runtime().get_io_pool().get_io_service().post(boost::bind
        (&output_stream::call_write_sync, this, in, id));

    // Sleep until the worker thread wakes us up.
    self.yield(threads::suspended);
} // }}}

}}}

