// macros.hpp

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

// Make HPX inspect tool happy: hpxinspect:nounnamed

// IMPORTANT : the JT28092007_macros_HPP_DEFINED needs to remain constant\
- don't change the macro name!
#ifndef JT28092007_macros_HPP_DEFINED
#define JT28092007_macros_HPP_DEFINED

/*
    VERY IMPORTANT:
    Not using #pragma once
    We might need to re-include this file, when defining the logs
*/

#include <hpx/util/logging/detail/fwd.hpp>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <hpx/util/logging/detail/log_keeper.hpp>

namespace hpx { namespace util { namespace logging {

#ifdef HPX_LOG_COMPILE_FAST_ON
#define HPX_LOG_COMPILE_FAST
#elif defined(HPX_LOG_COMPILE_FAST_OFF)
#undef HPX_LOG_COMPILE_FAST
#else
// by default, turned on
#define HPX_LOG_COMPILE_FAST
#endif


///////////////////////////////////////////////////////////////////////////////
// Defining filter Macros

/*
    when compile fast is "off", we always need HPX_LOG_MANIPULATE_LOGS,
    to get access to the logger class typedefs;
*/
#if !defined(HPX_LOG_COMPILE_FAST)
#if !defined(HPX_LOG_MANIPULATE_LOGS)

#define HPX_LOG_MANIPULATE_LOGS

#endif
#endif


#ifdef HPX_LOG_COMPILE_FAST
// ****** Fast compile ******

///////////////////////////////////////////////////////////////////////////////
// use log

#define HPX_USE_LOG_FIND_GATHER(name,log_type,base_type) \
    namespace hpx { namespace util { namespace logging { namespace log_define_ { \
        extern ::hpx::util::logging::detail::log_keeper< base_type, \
        name ## _HPX_LOG_impl_, ::hpx::util::logging::detail::call_write_logger_finder< \
        log_type, gather_msg > ::type > name ; \
    }}}} \
    using hpx { namespace util_log_define_:: name ;


#define HPX_USE_LOG(name,type) HPX_USE_LOG_FIND_GATHER(name, type, \
     ::hpx::util::logging::detail::fast_compile_with_default_gather<>::log_type )

#define HPX_DECLARE_LOG_KEEPER(name,log_type) \
    namespace hpx { namespace util { namespace logging { namespace log_declare_ { \
        extern ::hpx::util::logging::detail::log_keeper< log_type, \
         name ## _HPX_LOG_impl_ > name; \
    }}}} \
    using hpx { namespace util_log_declare_ :: name ;




#if !defined(HPX_LOG_MANIPULATE_LOGS)
// user is declaring logs
#define HPX_DECLARE_LOG_WITH_LOG_TYPE(name,log_type) \
    log_type& name ## _HPX_LOG_impl_(); \
    HPX_DECLARE_LOG_KEEPER(name,log_type)

#else
// user is defining logs
#define HPX_DECLARE_LOG_WITH_LOG_TYPE(name,log_type) \
    log_type& name ## _HPX_LOG_impl_(); \
    HPX_USE_LOG(name,log_type)

#endif


#define HPX_DECLARE_LOG_FIND_GATHER(name) \
    HPX_DECLARE_LOG_WITH_LOG_TYPE(name, \
    ::hpx::util::logging::detail::fast_compile_with_default_gather<>::log_type )

#define HPX_DECLARE_LOG(name,type) HPX_DECLARE_LOG_FIND_GATHER(name)

///////////////////////////////////////////////////////////////////////////////
// define log
#define HPX_DEFINE_LOG_FIND_GATHER(name, log_type, base_type, gather_msg) \
      base_type & name ## _HPX_LOG_impl_() \
{ typedef ::hpx::util::logging::detail::call_write_logger_finder< log_type, \
     gather_msg > ::type logger_type; \
  static logger_type i; return i; } \
    namespace { hpx::util::logging::detail::fake_using_log \
     ensure_log_is_created_before_main ## name ( name ## _HPX_LOG_impl_() ); } \
    namespace hpx { namespace util_log_declare_ { \
    ::hpx::util::logging::detail::log_keeper< base_type, name ## _HPX_LOG_impl_ > \
     name ; } \
    namespace hpx { namespace util_log_define_ { \
    ::hpx::util::logging::detail::log_keeper< base_type, \
     name ## _HPX_LOG_impl_, ::hpx::util::logging::detail::call_write_logger_finder< \
     log_type, gather_msg > ::type > name ; \
    } \
    using hpx { namespace util_log_define_ :: name ;

#define HPX_DEFINE_LOG(name,type) \
    HPX_DEFINE_LOG_FIND_GATHER(name, type, \
    ::hpx::util::logging::detail::fast_compile_with_default_gather<>::log_type, \
    ::hpx::util::logging::detail::fast_compile_with_default_gather<>::gather_msg )

#else
// don't compile fast

#define HPX_DECLARE_LOG(name,type) type& name ## _HPX_LOG_impl_(); \
    extern hpx::util::logging::detail::log_keeper<type, name ## _HPX_LOG_impl_ > name;
#define HPX_DEFINE_LOG(name,type)  type& name ## _HPX_LOG_impl_() \
    { static type i; return i; } \
    namespace { hpx::util::logging::detail::\
    fake_using_log ensure_log_is_created_before_main ## name \
    ( name ## _HPX_LOG_impl_() ); } \
    hpx::util::logging::detail::log_keeper<type, name ## _HPX_LOG_impl_ > name;

/**
    Advanced
*/
#define HPX_DECLARE_LOG_WITH_GATHER(name,type,gather_type) HPX_DECLARE_LOG(name,type)

#endif

///////////////////////////////////////////////////////////////////////////////
// Filter Macros

#define HPX_DECLARE_LOG_FILTER_NO_NAMESPACE_PREFIX(name,type) \
     type& name ## _HPX_LOG_filter_impl_(); \
     extern hpx::util::logging::detail::log_filter_keeper<type, \
     name ## _HPX_LOG_filter_impl_ > name;
#define HPX_DEFINE_LOG_FILTER_NO_NAMESPACE_PREFIX(name,type) \
      type& name ## _HPX_LOG_filter_impl_() \
    { static type i; return i; } \
    namespace { hpx::util::logging::detail::fake_using_log \
    ensure_log_is_created_before_main ## name ( name ## _HPX_LOG_filter_impl_() ); } \
    hpx::util::logging::detail::log_filter_keeper<type, \
    name ## _HPX_LOG_filter_impl_ > name;


#define HPX_DECLARE_LOG_FILTER(name,type) \
         HPX_DECLARE_LOG_FILTER_NO_NAMESPACE_PREFIX(name, ::hpx::util::logging:: type)

#define HPX_DEFINE_LOG_FILTER(name,type) \
         HPX_DEFINE_LOG_FILTER_NO_NAMESPACE_PREFIX(name, ::hpx::util::logging:: type)

///////////////////////////////////////////////////////////////////////////////
// Log Macros

#define HPX_LOG_USE_LOG(l, do_func, is_log_enabled) \
         if ( !(is_log_enabled) ) ; else l -> do_func

#define HPX_LOG_USE_LOG_IF_LEVEL(l, holder, the_level) \
         HPX_LOG_USE_LOG(l, read_msg().gather().out(), \
         holder->is_enabled(::hpx::util::logging::level:: the_level) )

#define HPX_LOG_USE_LOG_IF_FILTER(l, the_filter) \
         HPX_LOG_USE_LOG(l, read_msg().gather().out(), the_filter)

#define HPX_LOG_USE_SIMPLE_LOG_IF_FILTER(l, is_log_enabled) \
             if ( !(is_log_enabled) ) ; else l ->operator()

///////////////////////////////////////////////////////////////////////////////
// Format and Destination Macros

#define HPX_LOG_FORMAT_MSG(msg_class) \
    namespace hpx { namespace util { namespace logging { \
        template<> struct formatter::msg_type<override> { typedef msg_class & type; }; \
    }}}

#define HPX_LOG_DESTINATION_MSG(msg_class) \
    namespace hpx { namespace util { namespace logging { \
        template<> struct destination::msg_type<override> \
         { typedef const msg_class & type; }; \
    }}}

}}}

#endif

