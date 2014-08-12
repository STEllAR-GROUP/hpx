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
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/components/iostreams/server/output_stream.hpp>

#include <hpx/util/io_service_pool.hpp>

#include <iostream>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>

namespace hpx { namespace iostreams { namespace detail
{
    void buffer::save(hpx::util::portable_binary_oarchive & ar, unsigned) const
    {
        bool valid = (data_.get() && !data_->empty());
        ar & valid;
        if(valid)
        {
            ar & data_;
        }
    }

    void buffer::load(hpx::util::portable_binary_iarchive& ar, unsigned)
    {
        bool valid = false;
        ar & valid;
        if (valid)
        {
            ar & data_;
        }
    }
}}}

namespace hpx { namespace iostreams { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    void output_stream::call_write_async(boost::uint32_t locality_id,
        boost::uint64_t count, detail::buffer in)
    { // {{{
        // Perform the IO operation.
        pending_output_.output(locality_id, count, in, write_f, mtx_);
    } // }}}

    void output_stream::write_async(boost::uint32_t locality_id,
        boost::uint64_t count, detail::buffer const& in)
    { // {{{
        // Perform the IO in another OS thread.
        hpx::get_thread_pool("io_pool")->get_io_service().post(
            boost::bind(&output_stream::call_write_async, this, locality_id,
                count, in));
    } // }}}

    ///////////////////////////////////////////////////////////////////////////
    void output_stream::call_write_sync(boost::uint32_t locality_id,
        boost::uint64_t count, detail::buffer in, threads::thread_id_type caller)
    {
        // Perform the IO operation.
        pending_output_.output(locality_id, count, in, write_f, mtx_);

        // Wake up caller.
        threads::set_thread_state(caller, threads::pending);
    }

    void output_stream::write_sync(boost::uint32_t locality_id,
        boost::uint64_t count, detail::buffer const& in)
    { // {{{
        // Perform the IO in another OS thread.
        hpx::get_thread_pool("io_pool")->get_io_service().post(
            boost::bind(&output_stream::call_write_sync, this, locality_id,
                count, in, threads::get_self_id()));

        // Sleep until the worker thread wakes us up.
        this_thread::suspend(threads::suspended, "output_stream::write_sync");
    } // }}}
}}}

