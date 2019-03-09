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

#include <boost/current_function.hpp>
#include <hpx/util/logging/detail/cache_before_init_macros.hpp>
#include <string>

namespace hpx { namespace util { namespace logging {

/**
@page macros Macros - how, what for?

- @ref macros_if_else_strategy
- @ref macros_using
    - @ref macros_define_declare
        - @ref HPX_DECLARE_LOG
        - @ref HPX_DEFINE_LOG
        - @ref HPX_DECLARE_LOG_FILTER
        - @ref HPX_DEFINE_LOG_FILTER
    - @ref macros_use
        - @ref HPX_LOG_USE_LOG
        - @ref HPX_LOG_USE_LOG_IF_LEVEL
        - @ref HPX_LOG_USE_LOG_IF_FILTER
        - @ref HPX_LOG_USE_SIMPLE_LOG_IF_FILTER
    - @ref macros_compile_time
        - @ref macros_compile_time_fast
        - @ref macros_compile_time_slow
        - @ref HPX_LOG_compile_results


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
typedef logger_format_write logger_type;
HPX_DECLARE_LOG(g_l, logger_type)
@endcode


@subsubsection HPX_DEFINE_LOG HPX_DEFINE_LOG - defining a log

@code
HPX_DEFINE_LOG(log_name, logger_type, ...)
@endcode

This defines a log - and specifies some arguments to be used at its constructed.
It should be used in a source file, to define the log

Example:
@code
typedef logger_format_write logger_type;
...
HPX_DEFINE_LOG(g_l, logger_type)
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


@subsubsection HPX_DEFINE_LOG_FILTER HPX_DEFINE_LOG_FILTER
- defining a log filter with args


@code
HPX_DEFINE_LOG_FILTER(filter_name, filter_type, args)
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

Example:

typedef logger< destination::cout > app_log_type;

#define LAPP_ HPX_LOG_USE_SIMPLE_LOG_IF_FILTER(g_log_app(),
g_log_filter()->is_enabled() )
@endcode

See @ref defining_logger_macros for more details



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


*/

///////////////////////////////////////////////////////////////////////////////
// Defining filter Macros

#define HPX_DECLARE_LOG(name, type)                                           \
    type* name ();                                                            \
    namespace { void const* const ensure_creation_ ## name = name (); }

#define HPX_DEFINE_LOG(name, type, ...)                                       \
    type* name () { static type l { __VA_ARGS__ }; return &l; }


///////////////////////////////////////////////////////////////////////////////
// Filter Macros

#define HPX_DECLARE_LOG_FILTER(name, type)                                    \
    type* name ();                                                            \
    namespace { void const* const ensure_creation_ ## name = name (); }

#define HPX_DEFINE_LOG_FILTER(name, type, ...)                                \
    type* name () { static type l { __VA_ARGS__ }; return &l; }


///////////////////////////////////////////////////////////////////////////////
// Log Macros

#define HPX_LOG_USE_LOG_IF_LEVEL(l, holder, the_level)                        \
    HPX_LOG_USE_LOG(l, read_msg().gather().out(),                             \
        holder->is_enabled(::hpx::util::logging::level:: the_level) )

#define HPX_LOG_USE_LOG_IF_FILTER(l, the_filter)                              \
    HPX_LOG_USE_LOG(l, read_msg().gather().out(), the_filter)

#define HPX_LOG_USE_SIMPLE_LOG_IF_FILTER(l, is_log_enabled)                   \
    HPX_LOG_USE_LOG(l, read_msg().gather().out, is_log_enabled)


}}}

#endif
