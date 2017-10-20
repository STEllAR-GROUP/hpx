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

// IMPORTANT : the JT28092007_macros_HPP_DEFINED needs to remain constant
// - don't change the macro name!
#ifndef JT28092007_macros_HPP_DEFINED
#define JT28092007_macros_HPP_DEFINED

#if defined(HPX_MSVC) && (HPX_MSVC >= 1020)
# pragma once
#endif

#if !defined(HPX_LOG_TSS_USE_INTERNAL) && !defined(HPX_LOG_TSS_USE_BOOST) \
 && !defined(HPX_LOG_TSS_USE_CUSTOM) && !defined(HPX_HAVE_LOG_NO_TSS)
// use has not specified what TSS strategy to use
#define HPX_LOG_TSS_USE_INTERNAL

#endif

#include <boost/current_function.hpp>
#include <hpx/util/logging/detail/cache_before_init_macros.hpp>

namespace hpx { namespace util { namespace logging {

/**
@page macros Macros - how, what for?

- @ref macros_if_else_strategy
- @ref macros_using
    - @ref macros_define_declare
        - @ref HPX_DECLARE_LOG
        - @ref HPX_DEFINE_LOG
        - @ref HPX_DEFINE_LOG_WITH_ARGS
        - @ref HPX_DECLARE_LOG_FILTER
        - @ref HPX_DEFINE_LOG_FILTER
        - @ref HPX_DEFINE_LOG_FILTER_WITH_ARGS
    - @ref macros_use
        - @ref HPX_LOG_USE_LOG
        - @ref HPX_LOG_USE_LOG_IF_LEVEL
        - @ref HPX_LOG_USE_LOG_IF_FILTER
        - @ref HPX_LOG_USE_SIMPLE_LOG_IF_FILTER
    - @ref macros_set_formatters
        - @ref HPX_LOG_FORMAT_MSG
        - @ref HPX_LOG_DESTINATION_MSG
    - @ref macros_use_tags
        - @ref HPX_LOG_TAG
        - @ref HPX_LOG_TAG_LEVEL
        - @ref HPX_LOG_TAG_FILELINE
        - @ref HPX_LOG_TAG_FUNCTION
    - @ref macros_compile_time
        - @ref macros_compile_time_fast
        - @ref macros_compile_time_slow
        - @ref HPX_LOG_compile_results
    - @ref macros_tss
        - @ref HPX_LOG_TSS_USE_INTERNAL
        - @ref HPX_LOG_TSS_USE_BOOST
        - @ref HPX_LOG_TSS_USE_CUSTOM
        - @ref HPX_HAVE_LOG_NO_TSS







Simply put, you need to use macros to make sure objects (logger(s) and filter(s)) :
- are created before main
- are always created before being used

The problem we want to avoid is using a logger object before it's initialized
- this could happen
if logging from the constructor of a global/static object.

Using macros makes sure logging happens efficiently.
Basically what you want to achieve is something similar to:

@code
if ( is_filter_enabled)
    logger.gather_the_message_and_log_it();
@endcode



@section macros_if_else_strategy The if-else strategy

When gathering the message, what the macros will achieve is this:

@code
#define YOUR_COOL_MACRO_GOOD if ( !is_filter_enabled) \
; else logger.gather_the_message_and_log_it();
@endcode

The above is the correct way, instead of

@code
#define YOUR_COOL_MACRO_BAD if ( is_filter_enabled) \
logger.gather_the_message_and_log_it();
@endcode

because of

@code
if ( some_test)
  YOUR_COOL_MACRO_BAD << "some message ";
else
  whatever();
@endcode

In this case, @c whatever() will be called if @c some_test is true,
and if @c is_filter_enabled is false.

\n\n

@section macros_using Using the macros supplied with the library

There are several types of macros that this library supplies. They're explained below:

@subsection macros_define_declare Macros to declare/define logs/filters

@subsubsection HPX_DECLARE_LOG HPX_DECLARE_LOG - declaring a log

@code
HPX_DECLARE_LOG(log_name, logger_type)
@endcode

This declares a log. It should be used in a header file, to declare the log.
Note that @c logger_type only needs to be a declaration (a @c typedef, for instance)

Example:
@code
typedef logger_format_write< > logger_type;
HPX_DECLARE_LOG(g_l, logger_type)
@endcode


@subsubsection HPX_DEFINE_LOG HPX_DEFINE_LOG - defining a log

@code
HPX_DEFINE_LOG(log_name, logger_type)
@endcode

This defines a log. It should be used in a source file, to define the log.

Example:
@code
typedef logger_format_write< > logger_type;
...
HPX_DEFINE_LOG(g_l, logger_type)
@endcode


@subsubsection HPX_DEFINE_LOG_WITH_ARGS HPX_DEFINE_LOG_WITH_ARGS
- defining a log with arguments

@code
HPX_DEFINE_LOG_WITH_ARGS (log_name, logger_type, args)
@endcode

This defines a log - and specifies some arguments to be used at its constructed.
It should be used in a source file, to define the log.

Example:
@code
typedef logger< default_, destination::file> err_log_type;
...
HPX_DEFINE_LOG_WITH_ARGS( g_log_err(), err_log_type, ("err.txt") )
@endcode


@subsubsection HPX_DECLARE_LOG_FILTER HPX_DECLARE_LOG_FILTER - declaring a log filter

@code
HPX_DECLARE_LOG_FILTER(filter_name, filter_type)
@endcode

This declares a log filter.
It should be used in a header file, to declare the log filter.

Example:
@code
HPX_DECLARE_LOG_FILTER(g_log_filter, filter::no_ts )
@endcode


@subsubsection HPX_DEFINE_LOG_FILTER HPX_DEFINE_LOG_FILTER - defining a log filter

@code
HPX_DEFINE_LOG_FILTER(filter_name, filter_type)
@endcode

This defines a log filter. It should be used in a source file, to define the log filter.

Example:
@code
HPX_DEFINE_LOG_FILTER(g_log_filter, filter::no_ts )
@endcode



@subsubsection HPX_DEFINE_LOG_FILTER_WITH_ARGS HPX_DEFINE_LOG_FILTER_WITH_ARGS
- defining a log filter with args


@code
HPX_DEFINE_LOG_FILTER_WITH_ARGS(filter_name, filter_type, args)
@endcode

This defines a log filter - and specifies some arguments to be used at its constructed.
It should be used in a source file, to define the log filter.

Example:
@code
#define L_ HPX_DEFINE_LOG_FILTER(g_log_filter, filter::no_ts )
@endcode



@subsection macros_use Macros that help you define your own macros for logging

@subsubsection HPX_LOG_USE_LOG_IF_LEVEL HPX_LOG_USE_LOG_IF_LEVEL

Uses a logger if a filter has a certain level enabled:

@code
HPX_LOG_USE_LOG_IF_LEVEL(log, level_filter, level )
@endcode

Example:
@code
HPX_DECLARE_LOG_FILTER(g_log_level, hpx::util::logging::level::holder )
HPX_DECLARE_LOG(g_log_err, logger_type)

#define LERR_ HPX_LOG_USE_LOG_IF_LEVEL(g_log_err(), g_log_level(), error )
@endcode

See @ref defining_logger_macros for more details

@subsubsection HPX_LOG_USE_LOG_IF_FILTER HPX_LOG_USE_LOG_IF_FILTER

Uses a logger if a filter is enabled:

@code
HPX_LOG_USE_LOG_IF_FILTER(log, filter_is_enabled)
@endcode

Example:
@code
#define LERR_ HPX_LOG_USE_LOG_IF_FILTER(g_log_err(), g_log_filter()->is_enabled() )
@endcode

See @ref defining_logger_macros for more details

@subsubsection HPX_LOG_USE_LOG HPX_LOG_USE_LOG

Uses a logger:

@code
HPX_LOG_USE_LOG(l, do_func, is_log_enabled)
@endcode

Normally you don't use this directly.
You use @ref HPX_LOG_USE_LOG_IF_FILTER or @ref HPX_LOG_USE_LOG_IF_LEVEL instead.

See @ref defining_logger_macros for more details

@subsubsection HPX_LOG_USE_SIMPLE_LOG_IF_FILTER HPX_LOG_USE_SIMPLE_LOG_IF_FILTER

Uses a simple logger:

@code
HPX_LOG_USE_SIMPLE_LOG_IF_FILTER(l, is_log_enabled)
@endcode

A simple logger is one that uses a simple gather class (FIXME). Example:

@code
struct no_gather {
    const char * m_msg;
    no_gather() : m_msg(0) {}
    const char * msg() const { return m_msg; }
    void out(const char* msg) { m_msg = msg; }
    void out(const std::string& msg) { m_msg = msg.c_str(); }
};

typedef logger< no_gather, destination::cout > app_log_type;

#define LAPP_ HPX_LOG_USE_SIMPLE_LOG_IF_FILTER(g_log_app(),
g_log_filter()->is_enabled() )
@endcode

See @ref defining_logger_macros for more details



\n\n
@subsection macros_set_formatters Setting formatter/destination strings

@subsubsection HPX_LOG_FORMAT_MSG HPX_LOG_FORMAT_MSG

Sets the string class used by the formatter classes.
By default, it's <tt>std::(w)string</tt>

@code
HPX_LOG_FORMAT_MSG( string_class )
@endcode

You can do this to optimize formatting the message
- that is, use a string class optimized for appending and prepending messages
(which is basically what formatting is all about).

Example:
@code
HPX_LOG_FORMAT_MSG( optimize::cache_string_one_str<> )
@endcode


@subsubsection HPX_LOG_DESTINATION_MSG HPX_LOG_DESTINATION_MSG

Sets the string class used by the destination classes. By default,
it's <tt>std::(w)string</tt>

@code
HPX_LOG_DESTINATION_MSG( string_class )
@endcode

Example:
@code
HPX_LOG_DESTINATION_MSG( std::string )
@endcode

Usually you won't need to change this.
The destination classes don't change the contets of the string
- each class just writes the string
to a given destination.






\n\n

@subsection macros_use_tags Using tags

Note that tags are only used when you create your own macros for logging.
See the tag namespace.

@subsubsection HPX_LOG_TAG HPX_LOG_TAG

@code
HPX_LOG_TAG(tag_class)
@endcode

Adds a tag from the hpx::util::logging::tag namespace.
In other words, this is a shortcut for <tt> hpx::util::logging::tag::tag_class</tt>.
Note that in case the @c tag_class has a custom constructor,
you need to pass the params as well, after the macro, like shown below.

Example:

@code
#define L_(module_name) HPX_LOG_USE_LOG_IF_FILTER(g_l(), \
g_log_filter()->is_enabled() ) .set_tag( HPX_LOG_TAG(module)(module_name) )
@endcode

@subsubsection HPX_LOG_TAG_LEVEL HPX_LOG_TAG_LEVEL

Adds a level tag.

@code
HPX_LOG_TAG(tag_level)
@endcode

Example:

@code
#define LDBG_ HPX_LOG_USE_LOG_IF_LEVEL(g_log_dbg(), g_log_level(), debug ) \
.set_tag( HPX_LOG_TAG_LEVEL(debug) )
#define LERR_ HPX_LOG_USE_LOG_IF_LEVEL(g_log_dbg(), g_log_level(), error ) \
.set_tag( HPX_LOG_TAG_LEVEL(error) )
@endcode

@subsubsection HPX_LOG_TAG_FILELINE HPX_LOG_TAG_FILELINE

Ads the file/line tag (that is, the current @c __FILE__ and @c __LINE__ will be appended,
for each logged message).

@code
HPX_LOG_TAG_FILELINE
@endcode

Example:

@code
#define L_ HPX_LOG_USE_LOG_IF_FILTER(g_l(), g_log_filter()->is_enabled() \
) .set_tag( HPX_LOG_TAG_FILELINE)
@endcode

@subsubsection HPX_LOG_TAG_FUNCTION HPX_LOG_TAG_FUNCTION

Ads the function tag (that is, the @c BOOST_CURRENT_FUNCTION will be appended,
for each logged message).

@code
HPX_LOG_TAG_FUNCTION
@endcode

Example:

@code
#define L_ HPX_LOG_USE_LOG_IF_FILTER(g_l(), g_log_filter()->is_enabled() ) /
.set_tag( HPX_LOG_TAG_FUNCTION)
@endcode


\n\n

@subsection macros_compile_time Macros that treat compilation time

Assume you're using formatters and destinations, and you
<tt>#include <hpx/util/logging/format.hpp> </tt> in every
source file when you want to do logging.
This will increase compilation time quite a bit
(30 to 50%, in my tests; depending on your application' complexity,
this could go higher).

Thus, you can choose to:
-# have fast compilation time, and a virtual function call per
each logged message (default on debug mode)
-# have everything inline (no virtual function calls), very fast,
and slower compilation (default on release mode)

In the former case, most of the time you won't notice the extra virtual function call,
and the compilation time will be faster.

\n
@subsubsection macros_compile_time_fast Fast Compilation time

- this is turned on by default on debug mode
- this is turned off by default on release mode
- to force it, define HPX_LOG_COMPILE_FAST_ON directive
- applies only to logs that are @ref declare_define
"declared/defined using HPX_DECLARE_LOG and HPX_DEFINE_LOG macros"
  - this is @em transparent to you, the programmer
- to see what headers you should include, @ref headers_to_include "click here"


\n
@subsubsection macros_compile_time_slow Slow Compilation time

- this is turned off by default on debug mode
- this is turned on by default on release mode
- to force it, define HPX_LOG_COMPILE_FAST_OFF directive
- applies only to logs that are @ref declare_define
"declared/defined using HPX_DECLARE_LOG and HPX_DEFINE_LOG macros"
  - this is @em transparent to you, the programmer
- to see what headers you should include, @ref headers_to_include "click here"



\n
@subsubsection HPX_LOG_compile_results Compile time sample (and results)

Recently I created a sample (compile_time) to test the effect of
@c HPX_LOG_COMPILE_FAST_ON.
The results were not as promising as I had hoped. However, still,
when @c HPX_LOG_COMPILE_FAST_ON is on,
will compile faster by 30-40%. Noting that this is just an simple example,
the results might not be that conclusive.
Anyway, here they are:


Tested on 16 jan 2008/intel core duo 2.16Ghz machine, 5400Rpm HDD

- VC 8.0 (no precompiled header)
  - Debug
    - Compile with HPX_LOG_COMPILE_FAST_ON (default) - 33 secs
    - Compile with HPX_LOG_COMPILE_FAST_OFF  - 43 secs
- gcc 3.4.2
  - Debug
    - Compile with HPX_LOG_COMPILE_FAST_ON (default) - 24 secs
    - Compile with HPX_LOG_COMPILE_FAST_OFF  -  31 secs
- gcc 4.1
  - Debug
    - Compile with HPX_LOG_COMPILE_FAST_ON (default) - 20.5 secs
    - Compile with HPX_LOG_COMPILE_FAST_OFF  -  24 secs

If you have other results, or results from a big program using Boost Logging,
please share them with me. Thanks!







\n\n

@subsection macros_tss Macros that deal with Thread Specific Storage

These are the macros that specify what implementation of TSS (Thread Specific Storage)
we will be using.
Note that I did my best to remove the dependency on compat::thread
- the only dependence left is
when you use use a logger that writes everything @ref writer::on_dedicated_thread
"on a dedicated thread".

By default, for TSS, we use the internal implementation (no dependency).

The possibilities are:
- @ref HPX_LOG_TSS_USE_INTERNAL : use our internal implementation
  (no dependency on compat::thread)
- @ref HPX_LOG_TSS_USE_BOOST : use the implementation from compat::thread
  (dependency on compat::thread, of course).
- @ref HPX_LOG_TSS_USE_CUSTOM : uses a custom implementation.
  The interface of this implementation should match compat::thread's interface of
  @c thread_specific_ptr class
- @ref HPX_HAVE_LOG_NO_TSS : don't use TSS


@subsubsection HPX_LOG_TSS_USE_INTERNAL HPX_LOG_TSS_USE_INTERNAL

If defined, it uses our internal implementation for @ref macros_tss "TSS"

@subsubsection HPX_LOG_TSS_USE_BOOST HPX_LOG_TSS_USE_BOOST

If defined, it uses the compat::thread's implementation for @ref macros_tss "TSS"

@subsubsection HPX_LOG_TSS_USE_CUSTOM HPX_LOG_TSS_USE_CUSTOM

If defined, it uses a custom implementation for @ref macros_tss "TSS".
The interface of this implementation should match
compat::thread's interface of @c thread_specific_ptr class.

Your class should have this interface:
@code
template <typename T> class my_thread_specific_ptr ;
@endcode

When #defining HPX_LOG_TSS_USE_CUSTOM, do it like this:

@code
#define HPX_LOG_TSS_USE_CUSTOM = my_thread_specific_ptr
@endcode


@subsubsection HPX_HAVE_LOG_NO_TSS HPX_HAVE_LOG_NO_TSS

If defined, we don't use @ref macros_tss "TSS" as all.

*/

#ifdef HPX_LOG_COMPILE_FAST_ON
#define HPX_LOG_COMPILE_FAST
#elif defined(HPX_LOG_COMPILE_FAST_OFF)
#undef HPX_LOG_COMPILE_FAST
#else
    // turn it on otherwise
    #define HPX_LOG_COMPILE_FAST
#endif

///////////////////////////////////////////////////////////////////////////////
// Defining filter Macros

#ifdef HPX_LOG_COMPILE_FAST
// ****** Fast compile ******
#define HPX_DECLARE_LOG(name,type) \
     ::hpx::util::logging::logger_holder< type > & name (); \
    namespace { hpx::util::logging::ensure_early_log_creation \
     ensure_log_is_created_before_main ## name ( name () ); }

#define HPX_DEFINE_LOG(name,type) \
     ::hpx::util::logging::logger_holder< type > & name () \
    { static ::hpx::util::logging::logger_holder_by_value< type > l; return l; }

#define HPX_DEFINE_LOG_WITH_ARGS(name,type, args) \
     ::hpx::util::logging::logger_holder< type > & name () \
    { static ::hpx::util::logging::logger_holder_by_value< type > l \
     ( args ); return l; }

#else

// don't compile fast
#define HPX_DECLARE_LOG(name,type) type* name (); \
    namespace { hpx::util::logging::ensure_early_log_creation \
     ensure_log_is_created_before_main ## name ( * name () ); }

#define HPX_DEFINE_LOG(name,type)  type* name () \
    { static type l; return &l; }

#define HPX_DEFINE_LOG_WITH_ARGS(name,type, args)  type* name () \
    { static type l ( args ); return &l; }

#endif






///////////////////////////////////////////////////////////////////////////////
// Filter Macros

#define HPX_DECLARE_LOG_FILTER(name,type) type* name (); \
    namespace { hpx::util::logging::ensure_early_log_creation \
     ensure_log_is_created_before_main ## name ( * name () ); }

#define HPX_DEFINE_LOG_FILTER(name,type)  type * name () \
    { static type l; return &l; }

#define HPX_DEFINE_LOG_FILTER_WITH_ARGS(name,type, args)  type * name () \
    { static type l ( args ); return &l; }












///////////////////////////////////////////////////////////////////////////////
// Log Macros



#define HPX_LOG_USE_LOG_IF_LEVEL(l, holder, the_level) \
 HPX_LOG_USE_LOG(l, read_msg().gather().out(), \
 holder->is_enabled(::hpx::util::logging::level:: the_level) )

#define HPX_LOG_USE_LOG_IF_FILTER(l, the_filter) \
 HPX_LOG_USE_LOG(l, read_msg().gather().out(), the_filter)

#define HPX_LOG_USE_SIMPLE_LOG_IF_FILTER(l, is_log_enabled) \
 HPX_LOG_USE_LOG(l, read_msg().gather().out, is_log_enabled)




///////////////////////////////////////////////////////////////////////////////
// Format and Destination Macros

/** @section HPX_LOG_FORMAT_MSG HPX_LOG_FORMAT_MSG

@note
    When using HPX_LOG_FORMAT_MSG or HPX_LOG_DESTINATION_MSG,
    you must not be within any namespace scope.

    This is because when using this macro, as @c msg_class,
    you can specify any of your class, or
    something residing in @c hpx::util::logging namespace.
*/
#define HPX_LOG_FORMAT_MSG(msg_class) \
    namespace hpx { namespace util { namespace logging { namespace formatter { \
        template<> struct msg_type<override> { typedef msg_class type; }; \
    }}}}

/** @section HPX_LOG_DESTINATION_MSG HPX_LOG_DESTINATION_MSG

@note
    When using HPX_LOG_FORMAT_MSG or HPX_LOG_DESTINATION_MSG,
    you must not be within any namespace scope.

    This is because when using this macro, as @c msg_class,
    you can specify any of your class, or
    something residing in @c hpx::util::logging namespace.
*/
#define HPX_LOG_DESTINATION_MSG(msg_class) \
    namespace hpx { namespace util { namespace logging { namespace destination { \
        template<> struct msg_type<override> { typedef msg_class type; }; \
    }}}}







//////////////////////////////////////////////////////////////////////////////////////
// Tags

#define HPX_LOG_STRINGIZE2(x) #x
#define HPX_LOG_STRINGIZE(x) HPX_LOG_STRINGIZE2(x)
#define HPX_LOG_FILE_AND_LINE __FILE__ ":" HPX_LOG_STRINGIZE(__LINE__) " "


#define HPX_LOG_TAG(tag_type) ::hpx::util::logging::tag:: tag_type

#define HPX_LOG_TAG_LEVEL(lvl) HPX_LOG_TAG(level)(::hpx::util::logging::level ::lvl )

#define HPX_LOG_TAG_FILELINE HPX_LOG_TAG(file_line) (HPX_LOG_FILE_AND_LINE)

#define HPX_LOG_TAG_FUNCTION HPX_LOG_TAG(function) (BOOST_CURRENT_FUNCTION)


}}}

#endif

