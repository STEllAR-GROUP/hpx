////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <boost/ref.hpp>
#include <boost/bind.hpp>

#include <hpx/runtime.hpp>
#include <hpx/components/iostreams/server/output_stream.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>

namespace hpx { namespace iostreams { namespace server
{

void output_stream::call_write(
    std::deque<char> const& in
  , threads::thread_id_type id
) {
    // Perform the IO operation.
    write_f(in);

    // Wake up the calling px thread.
    threads::set_thread_state(id); 
}

void output_stream::write(std::deque<char> const& in)
{ // {{{
    mutex_type::scoped_lock l(mtx);

    // REVIEW: Should we confirm that the old thread_id has terminated
    // before we reset it?
    threads::thread_self& self = threads::get_self();
    threads::thread_id_type id = self.get_thread_id();

    // Perform the IO in another OS thread. 
    get_runtime().get_io_pool().get_io_service().post(boost::bind
        (&output_stream::call_write, this, boost::cref(in), id));

    // Yield this px thread. call_write will wake us up when the IO operation
    // is complete.
    self.yield(threads::suspended);
} // }}}

}}}

