////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>

#include <boost/ref.hpp>
#include <boost/bind.hpp>

#include <hpx/runtime.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/components/iostreams/server/output_stream.hpp>

#include <iostream>

namespace hpx { namespace iostreams { namespace server
{

///////////////////////////////////////////////////////////////////////////////
void output_stream::call_write_async(
    buffer const& in
) { // {{{
    mutex_type::scoped_lock l(mtx);

    // Perform the IO operation.
    write_f(*(in.data_));
} // }}}

void output_stream::write_async(
    buffer const& in
) { // {{{
    // Perform the IO in another OS thread.
    hpx::get_thread_pool("io_pool")->get_io_service().post(
        boost::bind(&output_stream::call_write_async, this, in));
} // }}}

///////////////////////////////////////////////////////////////////////////////
void output_stream::call_write_sync(
    buffer const& in
  , threads::thread_id_type caller
) {
    {
        mutex_type::scoped_lock l(mtx);

        // Perform the IO operation.
        write_f(*(in.data_));
    }

    // Wake up caller.
    threads::set_thread_state(caller, threads::pending);
}

void output_stream::write_sync(
    buffer const& in
) { // {{{
    threads::thread_self& self = threads::get_self();
    threads::thread_id_type id = self.get_thread_id();

    // Perform the IO in another OS thread.
    hpx::get_thread_pool("io_pool")->get_io_service().post(
        boost::bind(&output_stream::call_write_sync, this, in, id));

    // Sleep until the worker thread wakes us up.
    self.yield(threads::suspended);
} // }}}

}}}

