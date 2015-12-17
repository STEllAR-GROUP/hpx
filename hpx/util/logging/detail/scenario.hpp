// scenario.hpp

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

#ifndef JT28092007_format_fwd_HPP_DEFINED
#error Please include hpx/util/logging/format_fwd.hpp instead
#endif

#ifndef JT28092007_scenario_HPP_DEFINED
#define JT28092007_scenario_HPP_DEFINED

#if defined(HPX_MSVC) && (HPX_MSVC >= 1020)
# pragma once
#endif

#include <hpx/util/logging/detail/fwd.hpp>

namespace hpx { namespace util { namespace logging {
/** @page your_scenario_examples Examples of customizing your scenario

Example 1:
- Use a filter that uses per-thread caching with resync once at 10 secs,
- The filter uses levels
- Use a logger that will favor speed

@code
using namespace hpx::util::logging::scenario::usage;
typedef use< filter_::change::often<10>, filter_::level::use_levels, default_,
logger_::favor::speed> finder;

HPX_DECLARE_LOG_FILTER(g_log_filter, finder::filter);
HPX_DECLARE_LOG(g_l, finder::logger)
...
@endcode


Example 2:
- Use a filter that is initialized only once, when multiple threads are running
- The filter does not use levels
- Use a logger that is initialized only once, when only one thread is running

@code
using namespace hpx::util::logging::scenario::usage;
typedef use< filter_::change::set_once_when_multiple_threads,
filter_::level::no_levels, logger_::change::set_once_when_one_thread> finder;

HPX_DECLARE_LOG_FILTER(g_log_filter, finder::filter);
HPX_DECLARE_LOG(g_l, finder::logger)
...
@endcode

To see scenario::usage used in code:
- @ref common_your_scenario "Click to see description of the example"
- @ref common_your_scenario_code "Click to see the code"
*/

namespace filter {
    template<int> struct use_tss_with_cache ;
    struct no_ts ;
    struct ts ;
    struct use_tss_once_init ;
}

namespace level {
    template<int> struct holder_tss_with_cache ;
    struct holder_tss_once_init ;
    struct holder_ts;
    struct holder_no_ts ;
}
namespace writer {
    namespace threading {
        struct no_ts ;
        struct ts_write ;
        struct on_dedicated_thread ;
    }
}

/**
@brief Use this when you have a specific scenario,
and want the best logger/filter classes that fit that scenario.
Check out scenario::usage and scenario::ts.

For example, if you want to specify a %scenario based on usage:

@copydoc your_scenario_examples

*/
namespace scenario {

/**
@brief If you want the library to choose the best logger/filter
classes based on how your application will %use the loggers and filters,
%use this namespace.

First, don't forget to

@code
using namespace hpx::util::logging::scenario::usage;
@endcode

Then, you can specify the logger and filter, in a very easy manner

@copydoc your_scenario_examples

*/
namespace usage {
    // ... bring it in this namespace
    typedef ::hpx::util::logging::default_ default_;


    /** @brief Filter usage settings : filter_::change and filter_::level
    */
    namespace filter_ {
        /** @brief When does the filter change? */
        namespace change {
            /** @brief Optimize for %often %change. Does per-thread caching.
                At a given period, it re-synchronizes.

                This is the default, for a multi-threaded application.

                @param cache_period_secs At what period should we re-syncronize
            */
            template<int cache_period_secs = 5> struct often {};

            /** @brief Set only once, when there's only one thread running
            - thus, you don't need to worry about thread-syncronizing */
            struct set_once_when_one_thread {};

            /** @brief Set only once, when there could be multiple thread running.

            We automatically implement a strategy to check if
            the filter/logger has been initialized, and when it's done, we cache
            the result on every thread */
            struct set_once_when_multiple_threads {};

            /** @brief This is always accurate.
            However, it's the slowest too.

            In case of multiple threads,
            it always locks the logger/filter before accessing it.

            Not recommended,
            you should usually go with another strategy
            (often, set_once_when_one_thread or set_once_when_multiple_threads)
            */
            struct always_accurate {};

            /** @brief Single threading.
            It doesn't matter when/how %often the filter/logger changes.

                This is the default, for a single-threaded application.
            */
            struct single_thread {};

#ifndef HPX_HAVE_LOG_NO_TS
            typedef often<> default_;
#else
            typedef single_thread default_;
#endif
        }

        /** @brief What's our "level" policy? */
        namespace level {
            /** @brief not using levels (default) */
            struct no_levels {};
            /** @brief using levels */
            struct use_levels {};

            typedef no_levels default_;
        }
    }

    /** @brief Logger %usage settings : logger_::change and logger_::favor
    */
    namespace logger_ {
        /** @brief When does the logger change, that is, how often do you manipulate it?

        Note that using the log does not mean changing it.
        Manipulation means invoking non-const functions on the logger, like
        adding/removing formatters/destinations for instance.
        */
        namespace change {
            /** @brief Optimize for often change. Does per-thread caching.
            At a given period, it re-synchronizes.
            This is the default, for multi-threaded applications.

                @param cache_period_secs At what period should we re-syncronize
            */
            template<int cache_period_secs = 5> struct often {};

            /** @brief Set only once, when there's only one thread running
            - thus, you don't need to worry about thread-syncronizing */
            struct set_once_when_one_thread {};

            /** @brief Set only once, when there could be multiple thread running.

            We automatically implement a strategy to check if
            the filter/logger has been initialized, and when it's done, we cache
            the result on every thread */
            struct set_once_when_multiple_threads {};

            /** @brief This is always accurate. However, it's the slowest too.

            In case of multiple threads,
            it always locks the logger/filter before accessing it.

            Not recommended, you should usually go with another strategy
            (often, set_once_when_one_thread or set_once_when_multiple_threads)
            */
            struct always_accurate {};

            /** @brief Single threading.
            It doesn't matter when/how often the filter/logger changes.
            This is the default, for single-threaded applications.
            */
            struct single_thread {};

#ifndef HPX_HAVE_LOG_NO_TS
            typedef often<> default_;
#else
            typedef single_thread default_;
#endif
        }

        /** @brief When logging, what should we %favor? */
        namespace favor {
            /** @brief This will favor speed (logging will happen on a dedicated thread).
                The only problem you could have is if the application crashes.

                In this case, on Windows,
                the rest of the application will continue,
                and any non-flushed log message will be flushed.

                On POSIX, this may not be the case.
            */
            struct speed {};

            /** @brief All messages will be logged.
            This is the default for multi-threaded application
            */
            struct correctness {};

            /** @brief Single threading. It doesn't matter when/how often the
            filter/logger changes. This is the default, for single-threaded applications.
            */
            struct single_thread {};

#ifndef HPX_HAVE_LOG_NO_TS
            typedef correctness default_;
#else
            typedef single_thread default_;
#endif
        }

        /** @brief How do you gather the message? */
        namespace gather {
            /** @brief Using the cool operator<< (default) */
            struct ostream_like {};

            /** @brief If you want to use your custom class, specify it here */
            template<class gather_type> struct custom {};

            typedef ostream_like default_;
        }
    }



    namespace detail_find_filter {
        namespace level = ::hpx::util::logging::scenario::usage::filter_::level;
        namespace change = ::hpx::util::logging::scenario::usage::filter_::change;

        //////// use levels

        template<class change_> struct find_filter_use_levels {};

        template<int period_secs> struct find_filter_use_levels
            < change::often<period_secs> > {
            typedef ::hpx::util::logging::level::holder_tss_with_cache<period_secs> type;
        };

        template<> struct find_filter_use_levels
            < change::set_once_when_one_thread > {
            typedef ::hpx::util::logging::level::holder_no_ts type;
        };

        template<> struct find_filter_use_levels
            < change::set_once_when_multiple_threads > {
            typedef ::hpx::util::logging::level::holder_tss_once_init type;
        };

        template<> struct find_filter_use_levels< change::always_accurate > {
            typedef ::hpx::util::logging::level::holder_ts type;
        };

        template<> struct find_filter_use_levels< change::single_thread > {
            typedef ::hpx::util::logging::level::holder_no_ts type;
        };



        //////// no levels

        template<class change_> struct find_filter_no_levels {};

        template<int period_secs> struct find_filter_no_levels
            < change::often<period_secs> > {
            typedef ::hpx::util::logging::filter::use_tss_with_cache<period_secs> type;
        };

        template<> struct find_filter_no_levels
            < change::set_once_when_one_thread > {
            typedef ::hpx::util::logging::filter::no_ts type;
        };

        template<> struct find_filter_no_levels
            < change::set_once_when_multiple_threads >{
            typedef ::hpx::util::logging::filter::use_tss_once_init type;
        };

        template<> struct find_filter_no_levels< change::always_accurate > {
            typedef ::hpx::util::logging::filter::ts type;
        };

        template<> struct find_filter_no_levels< change::single_thread > {
            typedef ::hpx::util::logging::filter::no_ts type;
        };



        template<class change_, class level_> struct find_filter {
            // no levels
            typedef typename find_filter_no_levels<change_>::type type;
        };

        template<class change_> struct find_filter<change_, level::use_levels> {
            typedef typename find_filter_use_levels<change_>::type type;
        };

    }


    namespace detail_find_logger {
        namespace favor = ::hpx::util::logging::scenario::usage::logger_::favor;
        namespace change = ::hpx::util::logging::scenario::usage::logger_::change;
        namespace th = ::hpx::util::logging::writer::threading;
        namespace gather_usage = ::hpx::util::logging::scenario::usage::logger_::gather;

        template<class favor_> struct find_threading_from_favor {};
        template<> struct find_threading_from_favor<favor::speed>
        { typedef th::on_dedicated_thread type; };
        template<> struct find_threading_from_favor<favor::correctness>
        { typedef th::ts_write type; };
        template<> struct find_threading_from_favor<favor::single_thread>
        { typedef th::no_ts type; };

        template<class gather_type> struct find_gather {};
        template<> struct find_gather<gather_usage::ostream_like>
        { typedef ::hpx::util::logging::default_ type; };
        template<class custom_gather>
        struct find_gather<gather_usage::custom<custom_gather> >
        { typedef custom_gather type; };

        template<class favor_, class change_, class gather> struct find_logger {};

        template<class favor_, int period_secs, class gather>
        struct find_logger< favor_, change::often<period_secs>, gather > {
            typedef typename find_threading_from_favor<favor_>::type threading_type;
            template<int secs> struct lock_resource :
                ::hpx::util::logging::lock_resource_finder::tss_with_cache<secs> {};

            typedef ::hpx::util::logging::logger_format_write < default_,
                default_, threading_type, gather, lock_resource<period_secs> > type;
        };

        template<class favor_, class gather> struct find_logger< favor_,
            change::set_once_when_one_thread, gather > {
            typedef typename find_threading_from_favor<favor_>::type threading_type;
            typedef ::hpx::util::logging::lock_resource_finder
                ::single_thread lock_resource;

            typedef ::hpx::util::logging::logger_format_write< default_,
                default_, threading_type, gather, lock_resource> type;
        };

        template<class favor_, class gather> struct find_logger< favor_,
            change::set_once_when_multiple_threads, gather > {
            typedef typename find_threading_from_favor<favor_>
                ::type threading_type;
            typedef ::hpx::util::logging::lock_resource_finder
                ::tss_once_init<> lock_resource;

            typedef ::hpx::util::logging::logger_format_write< default_, default_,
                threading_type, gather, lock_resource> type;
        };

        template<class favor_, class gather> struct find_logger< favor_,
            change::always_accurate, gather > {
            typedef typename find_threading_from_favor<favor_>::type threading_type;
            typedef ::hpx::util::logging::lock_resource_finder::ts<> lock_resource;

            typedef ::hpx::util::logging::logger_format_write< default_, default_,
                threading_type, gather, lock_resource> type;
        };

        template<class favor_, class gather>
        struct find_logger< favor_, change::single_thread, gather > {
            typedef typename find_threading_from_favor<favor_>
                ::type threading_type;
            typedef ::hpx::util::logging::lock_resource_finder
                ::single_thread lock_resource;

            typedef ::hpx::util::logging::logger_format_write< default_, default_,
                threading_type, gather, lock_resource> type;
        };
    }

    /**
        @brief Finds a filter class and a logger class that fit your application's needs

        For this to happen, you will first need to specify your needs
        (the template parameters you'll pass to this class)

        @param filter_change @ref misc_use_defaults
        "(optional)" How does the %filter change?
        Any of the classes in the filter_::change namespace
        @param filter_level_ @ref misc_use_defaults
        "(optional)" Does our %filter %use levels?
        Any of the classes in the filter_::level namespace
        @param logger_change @ref misc_use_defaults
        "(optional)" How does our %logger change?
        Any of the classes in the logger_::change namespace
        @param logger_favor @ref misc_use_defaults
        "(optional)" What does the %logger favor?
        Any of the classes in the logger_::favor namespace
        @param logger_gather @ref misc_use_defaults
        "(optional)" What to %use as gather class.
        Any of the classes in the logger_::gather namespace

        @copydoc your_scenario_examples
    */
    template<
        class filter_change = default_,
        class filter_level = default_,
        class logger_change = default_,
        class logger_favor = default_,
        class logger_gather = default_ >
    struct use {

    private:
        typedef typename use_default<filter_change, filter_::change::default_ >
            ::type filter_change_type;
        typedef typename use_default<filter_level, filter_::level::default_ >
            ::type filter_level_type;

        typedef typename use_default<logger_change, logger_::change::default_ >
            ::type logger_change_type;
        typedef typename use_default<logger_favor, logger_::favor::default_>
            ::type logger_favor_type;
        typedef typename use_default<logger_gather, logger_::gather::default_>
            ::type gather_usage_type;

        typedef typename detail_find_logger::find_gather<gather_usage_type>
            ::type gather_type;

    public:
        typedef typename detail_find_filter::find_filter<filter_change_type,
            filter_level_type>::type filter;
        typedef typename detail_find_logger::find_logger< logger_favor_type,
            logger_change_type, gather_type>::type logger;

    };
}

/**
@brief Find out the right logger/filter, based on thread-safety of logger(s)/filter(s)

First, don't forget to

@code
using namespace hpx::util::logging::scenario::ts;
@endcode

Then, you can specify the logger and filter, in a very easy manner

Example:
- Use a filter that uses TSS (Thread Specific Storage)
- The filter uses levels
- Use a logger that uses TSS

@code
using namespace hpx::util::logging::scenario::ts;
typedef use< filter_::use_tss, level_::use_levels, logger_::use_tss> finder;

HPX_DECLARE_LOG_FILTER(g_log_filter, finder::filter);
HPX_DECLARE_LOG(g_l, finder::logger)
...
@endcode


To see how you can specify the logger/filter based on how you will %use them,
see usage namespace.
*/
namespace ts {
    // ... bring it in this namespace
    typedef ::hpx::util::logging::default_ default_;

    /** @brief filter uses levels? */
    struct level_ {
        /** @brief type of %filter levels %usage */
        enum type {
            /** @brief %use levels */
            use_levels,
            /** @brief don't %use levels */
            no_levels
        };
    };

    /** @brief filter thread-safety */
    struct filter_ {
        /** @brief type of filter thread-safety */
        enum type {
            /** @brief not thread-safe */
            none,
            /** @brief %use TSS (thread-specific storage) */
            use_tss,
            /** @brief thread-safe (but slow) */
            ts
        };
    };

    /** logger thread-safety */
    struct logger_ {
        /** @brief type of logger thread-safety */
        enum type {
            /** @brief not thread-safe */
            none,
            /** @brief %use TSS (thread-specific storage) */
            use_tss,
            /** @brief thread-safe (but slow) */
            ts
        };
    };

    namespace detail {
        namespace th = ::hpx::util::logging::writer::threading;

        template<filter_::type,level_::type> struct find_filter {};
        template<> struct find_filter<filter_::none, level_::no_levels >
        { typedef ::hpx::util::logging::filter::no_ts type; };
        template<> struct find_filter<filter_::use_tss, level_::no_levels>
        { typedef  ::hpx::util::logging::filter::use_tss_with_cache<5> type; };
        template<> struct find_filter<filter_::ts, level_::no_levels>
        { typedef ::hpx::util::logging::filter::ts type; };

        template<> struct find_filter<filter_::none, level_::use_levels >
        { typedef ::hpx::util::logging::level::holder_no_ts type; };
        template<> struct find_filter<filter_::use_tss, level_::use_levels >
        { typedef ::hpx::util::logging::level::holder_tss_with_cache<5> type; };
        template<> struct find_filter<filter_::ts, level_::use_levels >
        { typedef ::hpx::util::logging::level::holder_ts type; };

        template<logger_::type> struct find_logger {};
        template<> struct find_logger<logger_::none> {
            typedef ::hpx::util::logging::lock_resource_finder
                ::single_thread lock_resource;
            typedef ::hpx::util::logging::logger_format_write< default_,
                default_, th::no_ts, default_, lock_resource > type ;
        };
        template<> struct find_logger<logger_::use_tss> {
            typedef ::hpx::util::logging::lock_resource_finder::tss_with_cache<>
                lock_resource;

            typedef ::hpx::util::logging::logger_format_write< default_,
                default_, th::ts_write, default_, lock_resource > type ;
        };
        template<> struct find_logger<logger_::ts> {
            typedef ::hpx::util::logging::lock_resource_finder::ts<> lock_resource;

            typedef ::hpx::util::logging::logger_format_write< default_,
                default_, th::ts_write, default_, lock_resource > type ;
        };
    }

    /** @brief Find the right logger and filter,
         based on thread-safety: filter_::type, level_::type and logger_::type

        @copydoc ts
    */
    template<filter_::type filter_type, level_::type level_type,
        logger_::type logger_type> struct use {
        typedef typename detail::find_filter<filter_type,level_type>::type filter;
        typedef typename detail::find_logger<logger_type>::type logger;
    };
}

}

}}}

#endif

