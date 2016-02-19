// logger.hpp

// Boost Logging library
//
// Author: John Torjo, www.torjo.com
//
// Copyright (C) 2007 John Torjo (see www.torjo.com for email)
//
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org for updates, documentation, and revision history.
// See http://www.torjo.com/log2/ for more details

// IMPORTANT : the JT28092007_logger_HPP_DEFINED needs to remain constant
// - don't change the macro name!
#ifndef JT28092007_logger_HPP_DEFINED
#define JT28092007_logger_HPP_DEFINED

#if defined(HPX_MSVC) && (HPX_MSVC >= 1020)
# pragma once
#endif

#include <hpx/util/logging/detail/fwd.hpp>
#include <hpx/util/logging/detail/forward_constructor.hpp>
#include <hpx/util/logging/detail/find_gather.hpp>
#include <hpx/util/logging/detail/cache_before_init.hpp>
#include <boost/type_traits/remove_pointer.hpp>
#include <hpx/util/logging/detail/logger_base.hpp>

namespace hpx { namespace util { namespace logging {

    template<class holder, class gather_type> struct gather_holder { //-V690
        gather_holder(const holder & p_this) : m_this(p_this), m_use(true) {}

        gather_holder(const gather_holder & other) : m_this(other.m_this), m_use(true) {
            other.m_use = false;
        }

        ~gather_holder() {
            // FIXME handle exiting from exceptions!!!
            if ( m_use)
                m_this.on_do_write( detail::as_non_const(gather().msg()) );
        }
        gather_type & gather() { return m_obj; }
    private:
        const holder & m_this;
        mutable gather_type m_obj;
        mutable bool m_use;
    };


    template<class gather_msg = default_, class write_msg = default_ > struct logger ;


    // specialize when write_msg is not set
    // - in this case, you need to derive from this
    template<class gather_msg> struct logger<gather_msg, default_ >
        : logger_base<gather_msg, default_> {
        typedef typename detail::find_gather_if_default<gather_msg>
        ::gather_type gather_type;
        typedef typename gather_type::msg_type msg_type;
        typedef void_ write_type;

        typedef logger_base<gather_msg, default_> logger_base_type;
        using logger_base_type::cache;

        typedef logger<gather_msg, default_> self_type;

        logger() {}
        // we have virtual functions, lets have a virtual destructor as well
        // - many thanks Martin Baeker!
        virtual ~logger() {
            // force writing all messages from cache,
            // if cache hasn't been turned off yet
            turn_cache_off();
        }

        /**
            reads all data about a log message (gathers all the data about it)
        */
        gather_holder<self_type, gather_type> read_msg()
            const { return gather_holder<self_type, gather_type>(*this) ; }
//        gather_holder<self_type, gather_msg> read_msg()
//        const { return gather_holder<self_type, gather_msg>(*this) ; }

        write_type & writer()                    { return m_writer; }
        const write_type & writer() const        { return m_writer; }

        void turn_cache_off() {
            cache().turn_cache_off( call_do_write(*this) );
        }

    private:
        struct call_do_write {
            const logger & self_;
            call_do_write(const logger & s) : self_(s) {}
            void operator()(msg_type & msg) const {
                self_.do_write(msg);
            }
        };
    public:
        // called after all data has been gathered
        void on_do_write(msg_type & msg) const {
            if ( logger_base_type::is_cache_turned_off() )
                do_write(msg);
            else
                cache().add_msg(msg);
        }

        virtual void do_write(msg_type&) const = 0;
    private:
        // we don't know the writer
        void_ m_writer;
    };


    /**
        @brief Forwards everything to a different logger.

        This includes:
        - the writing (writer)
        - the caching
        - the on_destroyed (if present)
    */
    template<class gather_msg, class write_msg>
    struct forward_to_logger : logger<gather_msg, default_>
    {
        typedef typename detail::find_gather_if_default<gather_msg>
            ::gather_type gather_type;
        typedef typename gather_type::msg_type msg_type;
        typedef logger<gather_msg, default_> log_base;
        typedef typename log_base::cache_type cache_type;

        // ... might be called for a specialization of logger
        // - for logger<gather_msg,write_msg*>
        typedef typename boost::remove_pointer<write_msg>::type write_type;

        typedef logger<gather_msg, write_msg> original_logger_type;
        forward_to_logger(original_logger_type *original_logger = 0)
          : m_writer(0), m_original_logger( original_logger)
        {
            if ( m_original_logger)
                m_writer = &m_original_logger->writer();
        }

        /**
            specifies the logger to forward to
        */
        void forward_to(original_logger_type *original_logger) {
            m_original_logger = original_logger;
            m_writer = &m_original_logger->writer();
        }

        virtual void do_write(msg_type &a) const {
            (*m_writer)(a);
        }

        virtual cache_type & cache()     { return m_original_logger->cache(); }
        virtual const cache_type & cache() const
        { return m_original_logger->cache(); }

    private:
        write_type* m_writer;
        original_logger_type * m_original_logger;
    };



    /**
    @brief The logger class. Every log from your application is an instance of
    this (see @ref workflow_processing "workflow")

    As described in @ref workflow_processing "workflow",
    processing the message is composed of 2 things:
    - @ref workflow_2a "Gathering the message"
    - @ref workflow_2b "Processing the message"

    The logger class has 2 template parameters:


    @param gather_msg A new gather instance is created each time a message is written.
    The @c gather_msg class needs to be default-constructible.
    The @c gather_msg must have a function called @c .msg() which
    contains all information about the written message.
    It will be passed to the write_msg class.
    You can implement your own @c gather_msg class,
    or use any from the gather namespace.


    @param write_msg This is the object that does
    the @ref workflow_2b "second step" - the writing of the message.
    It can be a simple functor.
    Or, it can be a more complex object that contains
    logic of how the message is to be further formatted,
    and written to multiple destinations.
    You can implement your own @c write_msg class,
    or it can be any of the classes defined in writer namespace.
    Check out writer::format_write - which allows you to use
    several formatters to further format the message, and then write it to destinations.

    \n\n
    You will seldom need to use the logger class directly.
    You can use @ref defining_your_logger "other wrapper classes".


    \n\n
    The logger forwards
    the gathering of the message to the @c gather_msg class.
    Once all message is gathered, it's passed on to the writer.
    This is usually done through a @ref macros_use "macro".

    @code
    typedef logger< ... > logger_type;
    HPX_DECLARE_LOG_FILTER(g_log_filter, filter::no_ts )
    HPX_DECLARE_LOG(g_l, logger_type)

    #define L_ HPX_LOG_USE_LOG_IF_FILTER(g_l(), g_log_filter()->is_enabled() )

    // usage
    L_ << "this is so cool " << i++;

    @endcode



    \n\n
    To understand more on the workflow that involves %logging:
    - check out the gather namespace
    - check out the writer namespace

    */
    template<class gather_msg , class write_msg > struct logger
            // note: default implementation
            //- when gather_msg and write_msg are both known
            : logger_base<gather_msg, write_msg> {

        typedef typename detail::find_gather_if_default<gather_msg>
            ::gather_type gather_type;
        typedef typename gather_type::msg_type msg_type;
        typedef write_msg write_type;
        typedef logger_base<gather_msg, write_msg> logger_base_type;
        using logger_base_type::cache;

        typedef logger<gather_msg, write_msg> self_type;

        HPX_LOGGING_FORWARD_CONSTRUCTOR_INIT(logger,m_writer, init)

        ~logger() {
            // force writing all messages from cache,
            // if cache hasn't been turned off yet
            turn_cache_off();
        }

        /**
            reads all data about a log message (gathers all the data about it)
        */
        gather_holder<self_type, gather_type> read_msg()
            const { return gather_holder<self_type, gather_type>(*this) ; }

        write_msg & writer()                    { return m_writer; }
        const write_msg & writer() const        { return m_writer; }

        void turn_cache_off() {
            cache().turn_cache_off( writer() );
        }

        // called after all data has been gathered
        void on_do_write(msg_type & msg) const {
            if ( logger_base_type::is_cache_turned_off() )
                writer()(msg);
            else
                cache().add_msg(msg);
        }
    private:
        void init() {
            logger_base_type::m_base.forward_to(this);
        }

    private:
        write_msg m_writer;
        // a base object - one that can be used to log messages,
        // without having to know the full type of the log.
        forward_to_logger<gather_msg, write_type> m_base;
    };

    // specialize for write_msg* pointer!
    template<class gather_msg, class write_msg>
    struct logger<gather_msg, write_msg* > : logger_base<gather_msg, write_msg* > {
        typedef typename detail::find_gather_if_default<gather_msg>
        ::gather_type gather_type;
        typedef typename gather_type::msg_type msg_type;
        typedef write_msg write_type;

        typedef logger_base<gather_msg, write_msg* > logger_base_type;
        using logger_base_type::cache;


        typedef logger<gather_msg, write_msg*> self_type;

        logger(write_msg * writer_ = 0) : m_writer(writer_) {
            logger_base_type::m_base.forward_to(this);
        }
        ~logger() {
            // force writing all messages from cache,
            // if cache hasn't been turned off yet
            turn_cache_off();
        }

        void set_writer(write_msg* w) {
            m_writer = w;
        }

        /**
            reads all data about a log message (gathers all the data about it)
        */
        gather_holder<self_type, gather_msg> read_msg() const
        { return gather_holder<self_type, gather_msg>(*this) ; }

        write_msg & writer()                    { return *m_writer; }
        const write_msg & writer() const        { return *m_writer; }

        void turn_cache_off() {
            cache().turn_cache_off( writer() );
        }

        // called after all data has been gathered
        void on_do_write(msg_type & msg) const {
            if ( logger_base_type::is_cache_turned_off() )
                writer()(msg);
            else
                cache().add_msg(msg);
        }
    private:
        write_msg *m_writer;
        // a base object - one that can be used to log messages,
        // without having to know the full type of the log.
        forward_to_logger<gather_msg, write_type*> m_base;
    };




    /** @brief Given a logger class, finds its gather_msg,
    without needing to know the logger's definition (a typedef is enough)
    */
    template<class> struct logger_to_gather {};
    template<class gather_msg, class write_msg>
    struct logger_to_gather< logger<gather_msg,write_msg> > {
        typedef gather_msg gather_type;
    };

    /**

    @param write_msg the write message class. If a pointer, forwards to a pointer.
    If not a pointer, it holds it by value.
    */
    template<class gather_msg, class write_msg> struct implement_default_logger
        : logger<gather_msg, default_> {
        typedef typename gather_msg::msg_type msg_type;

        HPX_LOGGING_FORWARD_CONSTRUCTOR(implement_default_logger,m_writer)

        virtual void do_write(msg_type &a) const {
            m_writer(a);
        }

    private:
        write_msg m_writer;
    };

    // specialization for pointers
    template<class gather_msg, class write_msg>
    struct implement_default_logger<gather_msg,write_msg*>
        : logger<gather_msg, default_> {
        typedef typename gather_msg::msg_type msg_type;

        implement_default_logger(write_msg * writer = 0) : m_writer(writer) {
        }

        void set_writer(write_msg* writer) {
            m_writer = writer;
        }

        virtual void do_write(msg_type &a) const {
            (*m_writer)(a);
        }

    private:
        write_msg * m_writer;
    };

}}}

#endif

