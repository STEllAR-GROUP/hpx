//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_97FC0FA2_E773_4F83_8477_806EC68C2253)
#define HPX_97FC0FA2_E773_4F83_8477_806EC68C2253

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/local/recursive_mutex.hpp>

#include <iterator>
#include <ios>

#include <boost/swap.hpp>
#include <boost/noncopyable.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/atomic.hpp>

#include <hpx/state.hpp>
#include <hpx/include/client.hpp>
#include <hpx/components/iostreams/manipulators.hpp>
#include <hpx/components/iostreams/stubs/output_stream.hpp>
#include <hpx/util/move.hpp>

namespace hpx { namespace iostreams
{
    ///////////////////////////////////////////////////////////////////////////
    struct ostream;

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        // Tag types to be used to identify standard ostream objects
        struct cout_tag {};
        struct cerr_tag {};

        struct consolestream_tag {};

        ///////////////////////////////////////////////////////////////////////
        /// This is a Boost.IoStreams Sink that can be used to create an
        /// [io]stream on top of a detail::buffer.
        template <typename Char = char>
        struct buffer_sink
        {
            typedef Char char_type;

            struct category
              : boost::iostreams::sink_tag,
                boost::iostreams::flushable_tag
            {};

            explicit buffer_sink(ostream& os)
              : os_(os)
            {}

            // Write up to n characters to the underlying data sink into the
            // buffer s, returning the number of characters written.
            inline std::streamsize write(char_type const* s, std::streamsize n);

            // Make sure all content is sent to console
            inline bool flush();

        private:
            ostream& os_;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Char = char>
        struct ostream_creator
        {
            typedef std::back_insert_iterator<std::vector<Char> > iterator_type;
            typedef buffer_sink<Char> device_type;
            typedef boost::iostreams::stream<device_type> stream_type;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Tag>
        hpx::future<naming::id_type> create_ostream(Tag tag);

        void register_ostreams();
        void unregister_ostreams();
    }

    ///////////////////////////////////////////////////////////////////////////
    struct ostream
        : components::client_base<ostream, stubs::output_stream>
        , detail::buffer
        , detail::ostream_creator<char>::stream_type
    {
    private:
        typedef components::client_base<ostream, stubs::output_stream> base_type;
        typedef detail::ostream_creator<char>::stream_type stream_base_type;
        typedef detail::ostream_creator<char>::iterator_type iterator_type;
        typedef lcos::local::recursive_mutex mutex_type;

        HPX_MOVABLE_BUT_NOT_COPYABLE(ostream);

    private:
        mutex_type mtx_;
        boost::atomic<boost::uint64_t> generational_count_;

        // Performs a lazy streaming operation.
        template <typename T>
        ostream& streaming_operator_lazy(T const& subject)
        { // {{{
            // apply the subject to the local stream
            *static_cast<stream_base_type*>(this) << subject;
            return *this;
        } // }}}

        // Performs an asynchronous streaming operation.
        template <typename T, typename Lock>
        ostream& streaming_operator_async(T const& subject, Lock& l)
        { // {{{
            // apply the subject to the local stream
            *static_cast<stream_base_type*>(this) << subject;

            // If the buffer isn't empty, send it asynchronously to the
            // destination.
            if (!this->detail::buffer::empty())
            {
                // Create the next buffer, returns the previous buffer
                buffer next = this->detail::buffer::init();

                // Unlock the mutex before we cleanup.
                l.unlock();

                // Perform the write operation, then destroy the old buffer and
                // stream.
                this->base_type::write_async(get_gid(), hpx::get_locality_id(),
                    generational_count_++, next);
            }

            return *this;
        } // }}}

        // Performs a synchronous streaming operation.
        template <typename T, typename Lock>
        ostream& streaming_operator_sync(T const& subject, Lock& l)
        { // {{{
            // apply the subject to the local stream
            *static_cast<stream_base_type*>(this) << subject;

            // If the buffer isn't empty, send it to the destination.
            if (!this->detail::buffer::empty())
            {
                // Create the next buffer, returns the previous buffer
                buffer next = this->detail::buffer::init();

                // Unlock the mutex before we cleanup.
                l.unlock();

                // Perform the write operation, then destroy the old buffer and
                // stream.
                this->base_type::write_sync(get_gid(), hpx::get_locality_id(),
                    generational_count_++, next);
            }

            return *this;
        } // }}}

        ///////////////////////////////////////////////////////////////////////
        friend struct detail::buffer_sink<char>;

        bool flush()
        {
            mutex_type::scoped_lock l(mtx_);
            if (!this->detail::buffer::empty())
            {
                // Create the next buffer, returns the previous buffer
                buffer next = this->detail::buffer::init();

                // Unlock the mutex before we cleanup.
                l.unlock();

                // Perform the write operation, then destroy the old buffer and
                // stream.
                this->base_type::write_async(get_gid(), hpx::get_locality_id(),
                    generational_count_++, next);
            }
            return true;
        }

        ///////////////////////////////////////////////////////////////////////
        friend void detail::register_ostreams();
        friend void detail::unregister_ostreams();

        // late initialization during runtime system startup
        template <typename Tag>
        void initialize(Tag tag)
        {
            *static_cast<base_type*>(this) = detail::create_ostream(tag);
        }

        // reset this object during runtime system shutdown
        void uninitialize()
        {
            mutex_type::scoped_lock l(mtx_, boost::try_to_lock);
            if (l)
            {
                streaming_operator_sync(hpx::async_flush, l);   // unlocks l
            }
            this->base_type::free();
        }

    public:
        ostream()
          : base_type()
          , buffer()
          , stream_base_type(*this)
          , generational_count_(0)
        {}

        // hpx::flush manipulator
        ostream& operator<<(hpx::iostreams::flush_type const& m)
        {
            mutex_type::scoped_lock l(mtx_);
            return streaming_operator_sync(m, l);
        }

        // hpx::endl manipulator
        ostream& operator<<(hpx::iostreams::endl_type const& m)
        {
            mutex_type::scoped_lock l(mtx_);
            return streaming_operator_sync(m, l);
        }

        // hpx::async_flush manipulator
        ostream& operator<<(hpx::iostreams::async_flush_type const& m)
        {
            mutex_type::scoped_lock l(mtx_);
            return streaming_operator_async(m, l);
        }

        // hpx::async_endl manipulator
        ostream& operator<<(hpx::iostreams::async_endl_type const& m)
        {
            mutex_type::scoped_lock l(mtx_);
            return streaming_operator_async(m, l);
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        ostream& operator<<(T const& subject)
        {
            mutex_type::scoped_lock l(mtx_);
            return streaming_operator_lazy(subject);
        }

        using stream_base_type::operator<<;
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Char>
        inline std::streamsize buffer_sink<Char>::write(
            Char const* s, std::streamsize n)
        {
            return static_cast<buffer&>(os_).write(s, n);
        }

        template <typename Char>
        inline bool buffer_sink<Char>::flush()
        {
            return os_.flush();
        }
    }
}}

#endif // HPX_97FC0FA2_E773_4F83_8477_806EC68C2253

