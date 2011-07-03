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

#include <iostream>

namespace hpx { namespace iostreams { namespace server
{

void output_stream::call_write(
    std::deque<char> const& in
) {
//    std::cout << "trying lock" << std::endl;

    mutex_type::scoped_lock l(this);

    // Perform the IO operation.
    write_f(in);
}

void output_stream::write(std::deque<char> const& in)
{ // {{{
//    std::cout << "got " << in.size() << std::endl;

    // Perform the IO in another OS thread. 
    get_runtime().get_io_pool().get_io_service().post(boost::bind
        (&output_stream::call_write, this, in));
} // }}}

}}}

