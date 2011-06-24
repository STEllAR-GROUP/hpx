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

#include <hpx/runtime/components/client_base.hpp>
#include <hpx/components/iostreams/manipulators.hpp>
#include <hpx/components/iostreams/stubs/output_stream.hpp>
#include <hpx/util/iterator_sink.hpp>

// TODO: Optimize with move semantics (or smart pointers). Passing the out
// buffer to the lazy futures may lead to excessive copying.

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
    mutex_type mtx;
    std::deque<char> out_buffer;
    stream_type stream;

    // Performs a lazy, asynchronous streaming operation.
    template <typename T>
    lazy_ostream& streaming_operator_lazy(T const& subject)
    { // {{{
        mutex_type::scoped_lock(mtx);
        // apply the subject to the local stream
        stream << subject;
        return *this;
    } // }}}

    // Performs a synchronous streaming operation.
    template <typename T>
    lazy_ostream& streaming_operator_sync(T const& subject)
    { // {{{
        mutex_type::scoped_lock(mtx);

        // apply the subject to the local stream
        stream << subject;

        // If the buffer isn't empty, send it to the destination.
        if (!out_buffer.empty())
        {
            this->base_type::write_sync(gid_, out_buffer);
            out_buffer.clear();
        }

        return *this;
    } // }}}

  public:
    lazy_ostream(naming::id_type const& gid = naming::invalid_id)
        : base_type(gid)
        , stream(iterator_type(out_buffer))
    {} 

    ~lazy_ostream()
    { *this << hpx::flush; }

    // hpx::flush manipulator
    lazy_ostream& operator<<(hpx::iostreams::flush_type const& m)
    { return streaming_operator_sync(m); }

    // hpx::endl manipulator
    lazy_ostream& operator<<(hpx::iostreams::endl_type const& m)
    { return streaming_operator_sync(m); }

    template <typename T>
    lazy_ostream& operator<<(T const& subject)
    { return streaming_operator_lazy(subject); } 
};

}}

#endif // HPX_97FC0FA2_E773_4F83_8477_806EC68C2253

