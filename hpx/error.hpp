//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//  Copyright (c) 2014      Anuj R. Sharma
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file error.hpp

#if !defined(HPX_ERROR_SEP_08_2013_1109AM)
#define HPX_ERROR_SEP_08_2013_1109AM

#include <hpx/config.hpp>
#include <hpx/config/export_definitions.hpp>
#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Possible error conditions
    ///
    /// This enumeration lists all possible error conditions which can be
    /// reported from any of the API functions.
    enum error
    {
        success = 0,
        ///< The operation was successful
        no_success = 1,
        ///< The operation did failed, but not in an unexpected manner
        not_implemented = 2,
        ///< The operation is not implemented
        out_of_memory = 3,
        ///< The operation caused a out of memory condition
        bad_action_code = 4,
        ///<
        bad_component_type = 5,
        ///< The specified component type is not known or otherwise invalid
        network_error = 6,
        ///< A generic network error occurred
        version_too_new = 7,
        ///< The version of the network representation for this object is too new
        version_too_old = 8,
        ///< The version of the network representation for this object is too old
        version_unknown = 9,
        ///< The version of the network representation for this object is unknown
        unknown_component_address = 10,
        ///<
        duplicate_component_address = 11,
        ///< The given global id has already been registered
        invalid_status = 12,
        ///< The operation was executed in an invalid status
        bad_parameter = 13,
        ///< One of the supplied parameters is invalid
        internal_server_error = 14,
        ///<
        service_unavailable = 15,
        ///<
        bad_request = 16,
        ///<
        repeated_request = 17,
        ///<
        lock_error = 18,
        ///<
        duplicate_console = 19,
        ///< There is more than one console locality
        no_registered_console = 20,
        ///< There is no registered console locality available
        startup_timed_out = 21,
        ///<
        uninitialized_value = 22,
        ///<
        bad_response_type = 23,
        ///<
        deadlock = 24,
        ///<
        assertion_failure = 25,
        ///<
        null_thread_id = 26,
        ///< Attempt to invoke a API function from a non-HPX thread
        invalid_data = 27,
        ///<
        yield_aborted = 28,
        ///< The yield operation was aborted
        dynamic_link_failure = 29,
        ///<
        commandline_option_error = 30,
        ///< One of the options given on the command line is erroneous
        serialization_error = 31,
        ///< There was an error during serialization of this object
        unhandled_exception = 32,
        ///< An unhandled exception has been caught
        kernel_error = 33,
        ///< The OS kernel reported an error
        broken_task = 34,
        ///< The task associated with this future object is not available anymore
        task_moved = 35,
        ///< The task associated with this future object has been moved
        task_already_started = 36,
        ///< The task associated with this future object has already been started
        future_already_retrieved = 37,
        ///< The future object has already been retrieved
        promise_already_satisfied = 38,
        ///< The value for this future object has already been set
        future_does_not_support_cancellation = 39,
        ///< The future object does not support cancellation
        future_can_not_be_cancelled = 40,
        ///< The future can't be canceled at this time
        no_state = 41,
        ///< The future object has no valid shared state
        broken_promise = 42,
        ///< The promise has been deleted
        thread_resource_error = 43,
        ///<
        future_cancelled = 44,
        ///<
        thread_cancelled = 45,
        ///<
        thread_not_interruptable = 46,
        ///<
        duplicate_component_id = 47,
        ///< The component type has already been registered
        unknown_error = 48,
        ///< An unknown error occurred
        bad_plugin_type = 49,
        ///< The specified plugin type is not known or otherwise invalid
        security_error = 50,
        ///< An error occurred in the security component
        filesystem_error = 51,
        ///< The specified file does not exist or other filesystem related error
        bad_function_call = 52,
        ///< equivalent of std::bad_function_call
        task_canceled_exception = 53,
        ///< parallel::v2::task_canceled_exception
        task_block_not_active = 54,
        ///< task_region is not active
        out_of_range = 55,
        ///< Equivalent to std::out_of_range
        length_error = 56,
        ///< Equivalent to std::length_error

        /// \cond NOINTERNAL
        last_error,
        error_upper_bound = 0x7fffL   // force this enum type to be at least 16 bits.
        /// \endcond
    };

    /// \cond NOINTERNAL
    char const* const error_names[] =
    {
        /*  0 */ "success",
        /*  1 */ "no_success",
        /*  2 */ "not_implemented",
        /*  3 */ "out_of_memory",
        /*  4 */ "bad_action_code",
        /*  5 */ "bad_component_type",
        /*  6 */ "network_error",
        /*  7 */ "version_too_new",
        /*  8 */ "version_too_old",
        /*  9 */ "version_unknown",
        /* 10 */ "unknown_component_address",
        /* 11 */ "duplicate_component_address",
        /* 12 */ "invalid_status",
        /* 13 */ "bad_parameter",
        /* 14 */ "internal_server_error",
        /* 15 */ "service_unavailable",
        /* 16 */ "bad_request",
        /* 17 */ "repeated_request",
        /* 18 */ "lock_error",
        /* 19 */ "duplicate_console",
        /* 20 */ "no_registered_console",
        /* 21 */ "startup_timed_out",
        /* 22 */ "uninitialized_value",
        /* 23 */ "bad_response_type",
        /* 24 */ "deadlock",
        /* 25 */ "assertion_failure",
        /* 26 */ "null_thread_id",
        /* 27 */ "invalid_data",
        /* 28 */ "yield_aborted",
        /* 29 */ "dynamic_link_failure",
        /* 30 */ "commandline_option_error",
        /* 31 */ "serialization_error",
        /* 32 */ "unhandled_exception",
        /* 33 */ "kernel_error",
        /* 34 */ "broken_task",
        /* 35 */ "task_moved",
        /* 36 */ "task_already_started",
        /* 37 */ "future_already_retrieved",
        /* 38 */ "promise_already_satisfied",
        /* 39 */ "future_does_not_support_cancellation",
        /* 40 */ "future_can_not_be_cancelled",
        /* 41 */ "no_state",
        /* 42 */ "broken_promise",
        /* 43 */ "thread_resource_error",
        /* 44 */ "future_cancelled",
        /* 45 */ "thread_cancelled",
        /* 46 */ "thread_not_interruptable",
        /* 47 */ "duplicate_component_id",
        /* 48 */ "unknown_error",
        /* 49 */ "bad_plugin_type",
        /* 50 */ "security_error",
        /* 51 */ "filesystem_error",
        /* 52 */ "bad_function_call",
        /* 53 */ "task_canceled_exception",
        /* 54 */ "task_block_not_active",
        /* 55 */ "out_of_range",
        /* 56 */ "length_error",

        /*    */ ""
    };
    /// \endcond

    /// \cond NOINTERNAL

    /// \brief throw an hpx::exception initialized from the given arguments
    HPX_ATTRIBUTE_NORETURN HPX_EXPORT
    void throw_exception(error e, std::string const& msg,
        std::string const& func, std::string const& file = "", long line = -1);
    /// \endcond
}

#endif

