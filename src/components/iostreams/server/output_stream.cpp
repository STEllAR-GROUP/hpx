////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>
#include <hpx/util/bind.hpp>

#include <hpx/runtime_fwd.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/shared_ptr.hpp>
#include <hpx/runtime/serialization/vector.hpp>

#include <hpx/components/iostreams/server/buffer.hpp>
#include <hpx/components/iostreams/server/output_stream.hpp>

#include <hpx/util/io_service_pool.hpp>

#include <cstdint>
#include <memory>
#include <utility>

namespace hpx { namespace iostreams { namespace detail
{
    void buffer::save(serialization::output_archive & ar, unsigned) const
    {
        bool valid = (data_.get() && !data_->empty());
        ar & valid;
        if(valid)
        {
            ar & data_;
        }
    }

    void buffer::load(serialization::input_archive& ar, unsigned)
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
    void output_stream::call_write_async(std::uint32_t locality_id,
        std::uint64_t count, detail::buffer const& in)
    { // {{{
        // Perform the IO operation.
        pending_output_.output(locality_id, count, in, write_f, mtx_);
    } // }}}

    void output_stream::write_async(std::uint32_t locality_id,
        std::uint64_t count, detail::buffer const& buf_in)
    { // {{{
        // Perform the IO in another OS thread.
        detail::buffer in(buf_in);
        hpx::get_thread_pool("io_pool")->get_io_service().post(
            util::bind(&output_stream::call_write_async, this, locality_id,
                count, std::move(in)));
    } // }}}

    ///////////////////////////////////////////////////////////////////////////
    void output_stream::call_write_sync(std::uint32_t locality_id,
        std::uint64_t count, detail::buffer const& in,
        threads::thread_id_type caller)
    {
        // Perform the IO operation.
        pending_output_.output(locality_id, count, in, write_f, mtx_);

        // Wake up caller.
        threads::set_thread_state(caller, threads::pending);
    }

    void output_stream::write_sync(std::uint32_t locality_id,
        std::uint64_t count, detail::buffer const& buf_in)
    { // {{{
        // Perform the IO in another OS thread.
        detail::buffer in(buf_in);
        hpx::get_thread_pool("io_pool")->get_io_service().post(
            util::bind(&output_stream::call_write_sync, this, locality_id,
                count, std::ref(in), threads::get_self_id()));

        // Sleep until the worker thread wakes us up.
        this_thread::suspend(threads::suspended, "output_stream::write_sync");
    } // }}}
}}}
