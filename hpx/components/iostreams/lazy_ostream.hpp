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

#include <boost/noncopyable.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/move/move.hpp>

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

    typedef lcos::mutex mutex_type;
    typedef std::back_insert_iterator<std::deque<char> > iterator_type; 
    typedef util::iterator_sink<iterator_type> device_type;
    typedef boost::iostreams::stream<device_type> stream_type;
    // }}}

  private:
    BOOST_MOVABLE_BUT_NOT_COPYABLE(lazy_ostream)

    mutex_type mtx;

    struct data_type
    {
        std::deque<char> out_buffer;
        stream_type stream;

        data_type()
            : out_buffer(), stream(iterator_type(out_buffer)) {}
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

    // Performs a synchronous streaming operation.
    template <typename T>
    lazy_ostream& streaming_operator_sync(T const& subject)
    { // {{{
        // apply the subject to the local stream
        data->stream << subject;

        // If the buffer isn't empty, send it to the destination.
        if (!data->out_buffer.empty())
        {
            this->base_type::write_sync(gid_, data->out_buffer);
            data->out_buffer.clear();
        }

        return *this;
    } // }}}

  public:
    lazy_ostream(naming::id_type const& gid = naming::invalid_id)
        : base_type(gid)
        , data(new data_type)
    {} 

    lazy_ostream(BOOST_RV_REF(lazy_ostream) other)
    {
        BOOST_VERIFY(other.mtx.try_lock());
        BOOST_VERIFY(other.data);
        data = other.data;
        other.data = 0;
        other.mtx.unlock();
    }

    ~lazy_ostream()
    {
        if (threads::threadmanager_is(running))
        {
            mutex_type::scoped_lock l(mtx);
            if (data)   
            {
                streaming_operator_sync(hpx::flush);
                delete data;
                data = 0;
            }
        }
    }

    // hpx::flush manipulator
    lazy_ostream& operator<<(hpx::iostreams::flush_type const& m)
    {
        mutex_type::scoped_lock l(mtx);
        return streaming_operator_sync(m);
    }

    // hpx::endl manipulator
    lazy_ostream& operator<<(hpx::iostreams::endl_type const& m)
    {
        mutex_type::scoped_lock l(mtx);
        return streaming_operator_sync(m);
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

