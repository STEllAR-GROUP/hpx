//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_97FC0FA2_E773_4F83_8477_806EC68C2253)
#define HPX_97FC0FA2_E773_4F83_8477_806EC68C2253

#include <hpx/hpx_fwd.hpp>
#include <hpx/state.hpp>
#include <hpx/include/client.hpp>
#include <hpx/components/iostreams/manipulators.hpp>
#include <hpx/components/iostreams/stubs/output_stream.hpp>
#include <hpx/util/move.hpp>
#include <hpx/lcos/local/recursive_mutex.hpp>

#include <boost/swap.hpp>
#include <boost/noncopyable.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/atomic.hpp>
#include <boost/thread/locks.hpp>

#include <iterator>
#include <ios>
#include <iostream>

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
        inline std::ostream& get_outstream(cout_tag)
        {
            return std::cout;
        }

        inline std::ostream& get_outstream(cerr_tag)
        {
            return std::cerr;
        }

        std::stringstream& get_consolestream();

        inline std::ostream& get_outstream(consolestream_tag)
        {
            return get_consolestream();
        }

        inline char const* get_outstream_name(cout_tag)
        {
            return "/locality#console/output_stream#cout";
        }

        inline char const* get_outstream_name(cerr_tag)
        {
            return "/locality#console/output_stream#cerr";
        }

        inline char const* get_outstream_name(consolestream_tag)
        {
            return "/locality#console/output_stream#consolestream";
        }

        ///////////////////////////////////////////////////////////////////////
        hpx::future<naming::id_type>
        create_ostream(char const* name, std::ostream& strm);

        template <typename Tag>
        hpx::future<naming::id_type> create_ostream(Tag tag)
        {
            return create_ostream(get_outstream_name(tag), detail::get_outstream(tag));
        }

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
        typedef stream_base_type::traits_type stream_traits_type;
        typedef BOOST_IOSTREAMS_BASIC_OSTREAM(char, stream_traits_type) std_stream_type;
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
                this->base_type::write_async(get_id(), hpx::get_locality_id(),
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
                this->base_type::write_sync(get_id(), hpx::get_locality_id(),
                    generational_count_++, next);
            }

            return *this;
        } // }}}

        ///////////////////////////////////////////////////////////////////////
        friend struct detail::buffer_sink<char>;

        bool flush()
        {
            boost::unique_lock<mutex_type> l(mtx_);
            if (!this->detail::buffer::empty())
            {
                // Create the next buffer, returns the previous buffer
                buffer next = this->detail::buffer::init();

                // Unlock the mutex before we cleanup.
                l.unlock();

                // Perform the write operation, then destroy the old buffer and
                // stream.
                this->base_type::write_async(get_id(), hpx::get_locality_id(),
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
            boost::unique_lock<mutex_type> l(mtx_, boost::try_to_lock);
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
            boost::unique_lock<mutex_type> l(mtx_);
            return streaming_operator_sync(m, l);
        }

        // hpx::endl manipulator
        ostream& operator<<(hpx::iostreams::endl_type const& m)
        {
            boost::unique_lock<mutex_type> l(mtx_);
            return streaming_operator_sync(m, l);
        }

        // hpx::async_flush manipulator
        ostream& operator<<(hpx::iostreams::async_flush_type const& m)
        {
            boost::unique_lock<mutex_type> l(mtx_);
            return streaming_operator_async(m, l);
        }

        // hpx::async_endl manipulator
        ostream& operator<<(hpx::iostreams::async_endl_type const& m)
        {
            boost::unique_lock<mutex_type> l(mtx_);
            return streaming_operator_async(m, l);
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        ostream& operator<<(T const& subject)
        {
            boost::lock_guard<mutex_type> l(mtx_);
            return streaming_operator_lazy(subject);
        }

        ///////////////////////////////////////////////////////////////////////
        ostream& operator<<(std_stream_type& (*manip_fun)(std_stream_type&))
        {
            boost::lock_guard<mutex_type> l(mtx_);
            return streaming_operator_lazy(manip_fun);
        }
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

