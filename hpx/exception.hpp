//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_EXCEPTION_MAR_24_2008_0929AM)
#define HPX_EXCEPTION_MAR_24_2008_0929AM

#include <exception>
#include <string>
#include <iosfwd>

#include <hpx/config.hpp>
#include <hpx/hpx_fwd.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/filesystem_compatibility.hpp>

#include <boost/cstdint.hpp>
#include <boost/assert.hpp>
#include <boost/current_function.hpp>
#include <boost/throw_exception.hpp>
#include <boost/system/error_code.hpp>
#include <boost/system/system_error.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    enum error
    {
        success = 0,
        no_success = 1,
        not_implemented = 2,
        out_of_memory = 3,
        bad_action_code = 4,
        bad_component_type = 5,
        network_error = 6,
        version_too_new = 7,
        version_too_old = 8,
        version_unknown = 9,
        unknown_component_address = 10,
        duplicate_component_address = 11,
        invalid_status = 12,
        bad_parameter = 13,
        internal_server_error = 14,
        service_unavailable = 15,
        bad_request = 16,
        repeated_request = 17,
        lock_error = 18,
        duplicate_console = 19,
        no_registered_console = 20,
        startup_timed_out = 21,
        uninitialized_value = 22,
        bad_response_type = 23,
        deadlock = 24,
        assertion_failure = 25,
        null_thread_id = 26,
        invalid_data = 27,
        yield_aborted = 28,
        component_load_failure = 29,
        commandline_option_error = 30,
        serialization_error = 31,
        unhandled_exception = 32,
        kernel_error = 33,
        broken_task = 34,
        task_moved = 35,
        task_already_started = 36,
        future_already_retrieved = 38,
        future_already_satisfied = 39,
        future_does_not_support_cancellation = 40,
        future_can_not_be_cancelled = 41,
        future_uninitialized = 42,
        broken_promise = 43,
        thread_resource_error = 44,
        thread_interrupted = 45,
        thread_not_interruptable = 46,
        duplicate_component_id = 47,
        last_error,

        // force this enum type to be at least 16 bits.
        error_upper_bound = 0x7fffL
    };

    char const* const error_names[] =
    {
        "success",
        "no_success",
        "not_implemented",
        "out_of_memory",
        "bad_action_code",
        "bad_component_type",
        "network_error",
        "version_too_new",
        "version_too_old",
        "version_unknown",
        "unknown_component_address",
        "duplicate_component_address",
        "invalid_status",
        "bad_parameter",
        "internal_server_error",
        "service_unavailable",
        "bad_request",
        "repeated_request",
        "lock_error",
        "duplicate_console",
        "no_registered_console",
        "startup_timed_out",
        "uninitialized_value",
        "bad_response_type",
        "deadlock",
        "assertion_failure",
        "null_thread_id",
        "invalid_data",
        "yield_aborted",
        "component_load_failure",
        "commandline_option_error",
        "serialization_error",
        "unhandled_exception",
        "kernel_error",
        "broken_task",
        "task_moved",
        "task_already_started",
        "future_already_retrieved",
        "future_already_satisfied",
        "future_does_not_support_cancellation",
        "future_can_not_be_cancelled",
        "future_uninitialized",
        "broken_promise",
        "thread_resource_error",
        "thread_interrupted",
        "thread_not_interruptable",
        "duplicate_component_id",
        ""
    };

    namespace detail
    {
        class hpx_category : public boost::system::error_category
        {
        public:
            const char* name() const
            {
                return "HPX";
            }

            std::string message(int value) const
            {
                if (value >= success && value < last_error)
                    return std::string("HPX(") + error_names[value] + ")";

                return "HPX(unknown_error)";
            }
        };

        // this doesn't add any text to the exception what() message
        class hpx_category_rethrow : public boost::system::error_category
        {
        public:
            const char* name() const
            {
                return "";
            }

            std::string message(int) const
            {
                return "";
            }
        };
    } // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    //  Define the HPX error category
    inline boost::system::error_category const& get_hpx_category()
    {
        static detail::hpx_category instance;
        return instance;
    }

    inline boost::system::error_category const& get_hpx_rethrow_category()
    {
        static detail::hpx_category_rethrow instance;
        return instance;
    }

    ///////////////////////////////////////////////////////////////////////////
    enum throwmode
    {
        plain = 0,
        rethrow = 1
    };

    inline boost::system::error_code
    make_system_error_code(error e, throwmode mode = plain)
    {
        return boost::system::error_code(static_cast<int>(e),
            mode == rethrow ? get_hpx_rethrow_category() : get_hpx_category());
    }

    ///////////////////////////////////////////////////////////////////////////
    inline boost::system::error_condition
        make_error_condition(error e, throwmode mode)
    {
        return boost::system::error_condition(static_cast<int>(e),
            mode == rethrow ? get_hpx_rethrow_category() : get_hpx_category());
    }

    ///////////////////////////////////////////////////////////////////////////
    class error_code : public boost::system::error_code
    {
    public:
        explicit error_code(throwmode mode = plain)
          : boost::system::error_code(make_system_error_code(success, mode))
        {}

        explicit error_code(error e, char const* msg = "", throwmode mode = plain)
          : boost::system::error_code(make_system_error_code(e, mode))
          , message_(msg)
        {}

        error_code(error e, std::string const& msg, throwmode mode = plain)
          : boost::system::error_code(make_system_error_code(e, mode))
          , message_(msg)
        {}

        std::string const& get_message() const { return message_; }

    private:
        std::string message_;
    };

    inline error_code
    make_error_code(error e, throwmode mode = plain)
    {
        return error_code(e, "", mode);
    }

    inline error_code
    make_error_code(error e, char const* msg, throwmode mode = plain)
    {
        return error_code(e, msg, mode);
    }

    inline error_code
    make_error_code(error e, std::string const& msg, throwmode mode = plain)
    {
        return error_code(e, msg, mode);
    }

    inline error_code make_success_code(throwmode mode = plain)
    {
        return error_code(success, "success", mode);
    }

    ///////////////////////////////////////////////////////////////////////////
    class HPX_EXCEPTION_EXPORT exception : public boost::system::system_error
    {
    public:
        explicit exception(error e)
          : boost::system::system_error(make_error_code(e, plain))
        {
            BOOST_ASSERT(e >= success && e < last_error);
            LERR_(error) << "created exception: " << this->what();
        }
        explicit exception(boost::system::system_error const& e)
          : boost::system::system_error(e)
        {
            LERR_(error) << "created exception: " << this->what();
        }
        exception(error e, char const* msg, throwmode mode = plain)
          : boost::system::system_error(make_system_error_code(e, mode), msg)
        {
            BOOST_ASSERT(e >= success && e < last_error);
            LERR_(error) << "created exception: " << this->what();
        }
        exception(error e, std::string const& msg, throwmode mode = plain)
          : boost::system::system_error(make_system_error_code(e, mode), msg)
        {
            BOOST_ASSERT(e >= success && e < last_error);
            LERR_(error) << "created exception: " << this->what();
        }

        ~exception() throw()
        {
        }

        error get_error() const throw()
        {
            return static_cast<error>(
                this->boost::system::system_error::code().value());
        }

        error_code get_error_code(throwmode mode = plain) const throw()
        {
            return make_error_code(static_cast<error>(
                this->boost::system::system_error::code().value())
              , this->boost::system::system_error::what()
              , mode);
        }
    };

    namespace detail
    {
        struct HPX_EXCEPTION_EXPORT std_exception : std::exception
        {
          private:
            std::string what_;

          public:
            explicit std_exception(std::string const& w)
              : what_(w)
            {}

            ~std_exception() throw() {}

            const char* what() const throw()
            {
                return what_.c_str();
            }
        };

        struct HPX_EXCEPTION_EXPORT bad_alloc : std::bad_alloc
        {
          private:
            std::string what_;

          public:
            explicit bad_alloc(std::string const& w)
              : what_(w)
            {}

            ~bad_alloc() throw() {}

            const char* what() const throw()
            {
                return what_.c_str();
            }
        };

        struct HPX_EXCEPTION_EXPORT bad_exception : std::bad_exception
        {
          private:
            std::string what_;

          public:
            explicit bad_exception(std::string const& w)
              : what_(w)
            {}

            ~bad_exception() throw() {}

            const char* what() const throw()
            {
                return what_.c_str();
            }
        };

#ifndef BOOST_NO_TYPEID
        struct HPX_EXCEPTION_EXPORT bad_cast : std::bad_cast
        {
          private:
            std::string what_;

          public:
            explicit bad_cast(std::string const& w)
              : what_(w)
            {}

            ~bad_cast() throw() {}

            const char* what() const throw()
            {
                return what_.c_str();
            }
        };

        struct HPX_EXCEPTION_EXPORT bad_typeid : std::bad_typeid
        {
          private:
            std::string what_;

          public:
            explicit bad_typeid(std::string const& w)
              : what_(w)
            {}

            ~bad_typeid() throw() {}

            const char* what() const throw()
            {
                return what_.c_str();
            }
        };
#endif

        ///////////////////////////////////////////////////////////////////////
        // types needed to add additional information to the thrown exceptions
        struct tag_throw_locality {};
        struct tag_throw_hostname {};
        struct tag_throw_pid {};
        struct tag_throw_shepherd {};
        struct tag_throw_thread_id {};
        struct tag_throw_thread_name {};
        struct tag_throw_file {};
        struct tag_throw_function {};
        struct tag_throw_stacktrace {};
    }

    /// Stores the information about the locality id the exception has been
    /// raised on. This information will show up in error messages under the
    /// [locality] tag.
    typedef boost::error_info<detail::tag_throw_locality, boost::uint32_t>
        throw_locality;

    /// Stores the information about the hostname of the locality the exception
    /// has been raised on. This information will show up in error messages
    /// under the [hostname] tag.
    typedef boost::error_info<detail::tag_throw_hostname, std::string>
        throw_hostname;

    /// Stores the information about the pid of the OS process the exception
    /// has been raised on. This information will show up in error messages
    /// under the [pid] tag.
    typedef boost::error_info<detail::tag_throw_pid, boost::int64_t>
        throw_pid;

    /// Stores the information about the shepherd thread the exception has been
    /// raised on. This information will show up in error messages under the
    /// [shepherd] tag.
    typedef boost::error_info<detail::tag_throw_shepherd, std::size_t>
        throw_shepherd;

    /// Stores the information about the HPX thread the exception has been
    /// raised on. This information will show up in error messages under the
    /// [thread_id] tag.
    typedef boost::error_info<detail::tag_throw_thread_id, std::size_t>
        throw_thread_id;

    /// Stores the information about the HPX thread name the exception has been
    /// raised on. This information will show up in error messages under the
    /// [thread_name] tag.
    typedef boost::error_info<detail::tag_throw_thread_name, std::string>
        throw_thread_name;

    /// Stores the information about the function name the exception has been
    /// raised in. This information will show up in error messages under the
    /// [function] tag.
    typedef boost::error_info<detail::tag_throw_function, std::string>
        throw_function;

    /// Stores the information about the source file name the exception has
    /// been raised in. This information will show up in error messages under
    /// the [file] tag.
    typedef boost::error_info<detail::tag_throw_file, std::string>
        throw_file;

    /// Stores the information about the source file line number the exception
    /// has been raised at. This information will show up in error messages
    /// under the [line] tag.
    using boost::throw_line;

    /// Stores the information about the stack backtrace at the point the
    /// exception has been raised at. This information will show up in error
    /// messages under the [stack_trace] tag.
    typedef boost::error_info<detail::tag_throw_stacktrace, std::string>
        throw_stacktrace;

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        // rethrow an exception, internal helper
        template <typename Exception>
        BOOST_ATTRIBUTE_NORETURN HPX_EXPORT
        void rethrow_exception(Exception const& e,
            std::string const& func, std::string const& file, long line,
            std::string const& back_trace, boost::uint32_t node = 0,
            std::string const& hostname = "", boost::int64_t pid = -1,
            std::size_t shepherd = ~0, std::size_t thread_id = 0,
            std::string const& thread_name = "");

        // main function for throwing exceptions
        template <typename Exception>
        BOOST_ATTRIBUTE_NORETURN HPX_EXPORT
        void throw_exception(Exception const& e,
            std::string const& func, std::string const& file, long line);

        // BOOST_ASSERT handler
        BOOST_ATTRIBUTE_NORETURN HPX_EXPORT
        void assertion_failed(char const* expr, char const* function,
            char const* file, long line);

        // BOOST_ASSERT_MSG handler
        BOOST_ATTRIBUTE_NORETURN HPX_EXPORT
        void assertion_failed_msg(char const* msg, char const* expr,
            char const* function, char const* file, long line);

        // If backtrace support is enabled, this function returns the current
        // stack backtrace, otherwise it will return an empty string.
        HPX_EXPORT std::string backtrace();

        // Extract the diagnostic information embedded in the given exception and
        // return a string holding a formatted message.
        HPX_EXPORT std::string diagnostic_information(boost::exception const& e);
        HPX_EXPORT std::string diagnostic_information(boost::exception_ptr const& e);
        HPX_EXPORT std::string diagnostic_information(hpx::exception const& e);

        // Report an early or late exception and locally abort execution. There
        // isn't anything more we could do.
        HPX_EXPORT void report_exception_and_terminate(boost::exception_ptr const&);
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Extract the diagnostic information embedded in the given exception and
    /// return a string holding a formatted message.
    inline std::string diagnostic_information(boost::exception const& e)
    {
        return detail::diagnostic_information(e);
    }

    inline std::string diagnostic_information(boost::exception_ptr const& e)
    {
        return detail::diagnostic_information(e);
    }

    inline std::string diagnostic_information(hpx::exception const& e)
    {
        return detail::diagnostic_information(e);
    }
}

///////////////////////////////////////////////////////////////////////////////
namespace boost
{
    // forwarder for BOOST_ASSERT handler
    inline void assertion_failed(char const* expr, char const* function,
        char const* file, long line)
    {
        hpx::detail::assertion_failed(expr, function, file, line);
    }

    // forwarder for BOOST_ASSERT_MSG handler
    inline void assertion_failed_msg(char const* msg, char const* expr,
        char const* function, char const* file, long line)
    {
        hpx::detail::assertion_failed_msg(msg, expr, function, file, line);
    }

    namespace system
    {
        // make sure our errors get recognized by the Boost.System library
        template<> struct is_error_code_enum<hpx::error>
        {
            static const bool value = true;
        };

        template<> struct is_error_condition_enum<hpx::error>
        {
            static const bool value = true;
        };
    }
}

///////////////////////////////////////////////////////////////////////////////
// helper macro allowing to prepend file name and line number to a generated
// exception
#define HPX_THROW_EXCEPTION_EX(except, errcode, func, msg, mode)              \
    {                                                                         \
        boost::filesystem::path p__(hpx::util::create_path(__FILE__));        \
        hpx::detail::throw_exception(except(static_cast<hpx::error>(errcode), msg, mode),  \
            func, p__.string(), __LINE__);                                    \
    }                                                                         \
    /**/

#define HPX_THROW_STD_EXCEPTION(except, func)                                 \
    {                                                                         \
        boost::filesystem::path p__(hpx::util::create_path(__FILE__));        \
        hpx::detail::throw_exception(except, func, p__.string(), __LINE__);   \
    }                                                                         \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_THROW_EXCEPTION(errcode, f, msg)                                  \
    HPX_THROW_EXCEPTION_EX(hpx::exception, errcode, f, msg, hpx::plain)       \
    /**/

#define HPX_RETHROW_EXCEPTION(errcode, f, msg)                                \
    HPX_THROW_EXCEPTION_EX(hpx::exception, errcode, f, msg, hpx::rethrow)     \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_THROWS_IF(ec, errcode, f, msg)                                    \
    {                                                                         \
        if (&ec == &hpx::throws) {                                            \
            HPX_THROW_EXCEPTION(errcode, f, msg);                             \
        } else {                                                              \
            ec = make_error_code(static_cast<hpx::error>(errcode), msg);      \
        }                                                                     \
    }                                                                         \
    /**/

#define HPX_RETHROWS_IF(ec, errcode, f, msg)                                  \
    {                                                                         \
        if (&ec == &hpx::throws) {                                            \
            HPX_RETHROW_EXCEPTION(errcode, f, msg);                           \
        } else {                                                              \
            ec = make_error_code(static_cast<hpx::error>(errcode), msg);      \
        }                                                                     \
    }                                                                         \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_THROW_IN_CURRENT_FUNC(errcode, msg)                               \
    HPX_THROW_EXCEPTION(errcode, BOOST_CURRENT_FUNCTION, msg)                 \
    /**/

#define HPX_RETHROW_IN_CURRENT_FUNC(errcode, msg)                             \
    HPX_RETHROW_EXCEPTION(errcode, BOOST_CURRENT_FUNCTION, msg)               \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_THROWS_IN_CURRENT_FUNC_IF(ec, errcode, msg)                       \
    {                                                                         \
        if (&ec == &hpx::throws) {                                            \
            HPX_THROW_EXCEPTION(errcode, BOOST_CURRENT_FUNCTION, msg);        \
        } else {                                                              \
            ec = make_error_code(static_cast<hpx::error>(errcode), msg);      \
        }                                                                     \
    }                                                                         \
    /**/

#define HPX_RETHROWS_IN_CURRENT_FUNC_IF(ec, errcode, msg)                     \
    {                                                                         \
        if (&ec == &hpx::throws) {                                            \
            HPX_RETHROW_EXCEPTION(errcode, f, msg);                           \
        } else {                                                              \
            ec = make_error_code(static_cast<hpx::error>(errcode), msg);      \
        }                                                                     \
    }                                                                         \
    /**/

#include <hpx/config/warnings_suffix.hpp>

#endif

