////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_97FC0FA2_E773_4F83_8477_806EC68C2253)
#define HPX_97FC0FA2_E773_4F83_8477_806EC68C2253

#include <iterator>
#include <ios>

#include <boost/swap.hpp>
#include <boost/noncopyable.hpp>
#include <boost/iostreams/stream.hpp>

#include <hpx/state.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/components/iostreams/manipulators.hpp>
#include <hpx/components/iostreams/stubs/output_stream.hpp>
#include <hpx/util/iterator_sink.hpp>

namespace hpx { namespace iostreams
{

struct lazy_ostream
    : components::client_base<lazy_ostream, stubs::output_stream>
    , boost::noncopyable
{
    // {{{ types
    typedef components::client_base<lazy_ostream, stubs::output_stream>
        base_type;

    typedef lcos::local::mutex mutex_type;
    typedef std::back_insert_iterator<std::deque<char> > iterator_type;
    typedef util::iterator_sink<iterator_type> device_type;
    typedef boost::iostreams::stream<device_type> stream_type;
    // }}}

  private:
    mutex_type mtx;

    struct data_type
    {
        buffer out_buffer;
        stream_type stream;

        data_type()
          : out_buffer(new std::deque<char>)
          , stream(iterator_type(*(out_buffer.data_))) {}
    };

    data_type* data;

    // Performs a lazy streaming operation.
    template <typename T>
    lazy_ostream& streaming_operator_lazy(T const& subject)
    { // {{{
        // apply the subject to the local stream
        data->stream << subject;
        return *this;
    } // }}}

    // Performs an asynchronous streaming operation.
    template <typename T, typename Lock>
    lazy_ostream& streaming_operator_async(T const& subject, Lock& l)
    { // {{{
        // apply the subject to the local stream
        data->stream << subject;

        // If the buffer isn't empty, send it to the destination.
        if (!data->out_buffer.data_->empty())
        {
            // Create the next buffer/stream.
            data_type* next = new data_type;

            // Swap the current buffer for the next one.
            boost::swap(next, data);

            // Perform the write operation, then destroy the old buffer and
            // stream.
            this->base_type::write_async(gid_, next->out_buffer);

            // Unlock the mutex before we cleanup.
            mtx.unlock();

            delete next;
            next = 0;
        }

        return *this;
    } // }}}

    // Performs a synchronous streaming operation.
    template <typename T, typename Lock>
    lazy_ostream& streaming_operator_sync(T const& subject, Lock& l)
    { // {{{
        // apply the subject to the local stream
        data->stream << subject;

        // If the buffer isn't empty, send it to the destination.
        if (!data->out_buffer.data_->empty())
        {
            // Create the next buffer/stream.
            data_type* next = new data_type;

            // Swap the current buffer for the next one.
            boost::swap(next, data);

            // Perform the write operation, then destroy the old buffer and
            // stream.
            this->base_type::write_sync(gid_, next->out_buffer);

            // Unlock the mutex before we cleanup.
            l.unlock();

            delete next;
            next = 0;
        }

        return *this;
    } // }}}

  public:
    lazy_ostream(naming::id_type const& gid = naming::invalid_id)
      : base_type(gid)
      , data(new data_type)
    {}

    ~lazy_ostream()
    {
        if (threads::threadmanager_is(running))
        {
            mutex_type::scoped_lock l(mtx, boost::try_to_lock);
            if (l && data)
            {
                streaming_operator_sync(hpx::sync_flush, l);
            }

            if (data)
            {
                delete data;
                data = 0;
            }
        }
    }

    // hpx::flush manipulator (alias for hpx::sync_flush)
    lazy_ostream& operator<<(hpx::iostreams::flush_type const& m)
    {
        mutex_type::scoped_lock l(mtx);
        return streaming_operator_sync(m, l);
    }

    // hpx::endl manipulator (alias for hpx::sync_endl)
    lazy_ostream& operator<<(hpx::iostreams::endl_type const& m)
    {
        mutex_type::scoped_lock l(mtx);
        return streaming_operator_sync(m, l);
    }

    // hpx::sync_flush manipulator
    lazy_ostream& operator<<(hpx::iostreams::sync_flush_type const& m)
    {
        mutex_type::scoped_lock l(mtx);
        return streaming_operator_sync(m, l);
    }

    // hpx::sync_endl manipulator
    lazy_ostream& operator<<(hpx::iostreams::sync_endl_type const& m)
    {
        mutex_type::scoped_lock l(mtx);
        return streaming_operator_sync(m, l);
    }

    // hpx::async_flush manipulator
    lazy_ostream& operator<<(hpx::iostreams::async_flush_type const& m)
    {
        mutex_type::scoped_lock l(mtx);
        return streaming_operator_async(m, l);
    }

    // hpx::async_endl manipulator
    lazy_ostream& operator<<(hpx::iostreams::async_endl_type const& m)
    {
        mutex_type::scoped_lock l(mtx);
        return streaming_operator_async(m, l);
    }

    template <typename T>
    lazy_ostream& operator<<(T const& subject)
    {
        mutex_type::scoped_lock l(mtx);
        return streaming_operator_lazy(subject);
    }
};

}}

#endif // HPX_97FC0FA2_E773_4F83_8477_806EC68C2253

