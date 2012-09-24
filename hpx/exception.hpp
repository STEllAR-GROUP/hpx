//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file exception.hpp

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
    /// \cond NOINTERNAL
    // forward declaration
    class error_code;
    class exception;
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Generic error_conditions
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
        dynamic_link_failure = 29,
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

        /// \cond NOINTERNAL
        last_error,
        error_upper_bound = 0x7fffL   // force this enum type to be at least 16 bits.
        /// \endcond
    };

    /// \cond NOINTERNAL
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
        "dynamic_link_failure",
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
    /// \endcond

    /// \cond NODETAIL
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

        struct lightweight_hpx_category : hpx_category {};

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

        struct lightweight_hpx_category_rethrow : hpx_category_rethrow {};

        ///////////////////////////////////////////////////////////////////////
        boost::exception_ptr access_exception(error_code const&);

    } // namespace detail
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns generic HPX error category used for new errors.
    inline boost::system::error_category const& get_hpx_category()
    {
        static detail::hpx_category instance;
        return instance;
    }

    /// \brief Returns generic HPX error category used for errors re-thrown
    ///        after the exception has been de-serialized.
    inline boost::system::error_category const& get_hpx_rethrow_category()
    {
        static detail::hpx_category_rethrow instance;
        return instance;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Encode error category for new error_code.
    enum throwmode
    {
        plain = 0,
        rethrow = 1,
        lightweight = 0x80, // do not generate an exception for this error_code
        /// \cond NODETAIL
        lightweight_rethrow = lightweight | rethrow
        /// \endcond
    };

    /// \cond NOINTERNAL
    inline boost::system::error_category const& 
    get_lightweight_hpx_category()
    {
        static detail::lightweight_hpx_category instance;
        return instance;
    }

    inline boost::system::error_category const& get_hpx_category(throwmode mode)
    {
        switch(mode) {
        case rethrow:
            return get_hpx_rethrow_category();

        case lightweight:
        case lightweight_rethrow:
            return get_lightweight_hpx_category();

        case plain:
        default:
            break;
        }
        return get_hpx_category();
    }

    inline boost::system::error_code
    make_system_error_code(error e, throwmode mode = plain)
    {
        
        return boost::system::error_code(
            static_cast<int>(e), get_hpx_category(mode));
    }

    ///////////////////////////////////////////////////////////////////////////
    inline boost::system::error_condition
    make_error_condition(error e, throwmode mode)
    {
        return boost::system::error_condition(
            static_cast<int>(e), get_hpx_category(mode));
    }
    /// \endcond

    /// \cond NODETAIL
    namespace detail
    {
        // main function for throwing exceptions
        template <typename Exception>
        HPX_EXPORT boost::exception_ptr
            get_exception(Exception const& e,
                std::string const& func = "<unknown>",
                std::string const& file = "<unknown>",
                long line = -1);
    }
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    /// \brief A hpx::error_code represents an arbitrary error condition.
    ///
    /// The class hpx::error_code describes an object used to hold error code
    /// values, such as those originating from the operating system or other
    /// low-level application program interfaces.
    ///
    /// \note Class hpx::error_code is an adjunct to error reporting by
    /// exception
    ///
    class error_code : public boost::system::error_code
    {
    public:
        /// Construct an object of type error_code.
        ///
        /// \param mode   The parameter \p mode specifies whether the constructed
        ///               hpx::error_code belongs to the error category
        ///               \a hpx_category (if mode is \a plain, this is the
        ///               default) or to the category \a hpx_category_rethrow
        ///               (if mode is \a rethrow).
        ///
        /// \throws nothing
        explicit error_code(throwmode mode = plain)
          : boost::system::error_code(make_system_error_code(success, mode))
        {}

        /// Construct an object of type error_code.
        ///
        /// \param e      The parameter \p e holds the hpx::error code the new
        ///               exception should encapsulate.
        /// \param mode   The parameter \p mode specifies whether the constructed
        ///               hpx::error_code belongs to the error category
        ///               \a hpx_category (if mode is \a plain, this is the
        ///               default) or to the category \a hpx_category_rethrow
        ///               (if mode is \a rethrow).
        ///
        /// \throws nothing
        inline explicit error_code(error e, throwmode mode = plain);

        /// Construct an object of type error_code.
        ///
        /// \param e      The parameter \p e holds the hpx::error code the new
        ///               exception should encapsulate.
        /// \param func   The name of the function where the error was raised.
        /// \param file   The file name of the code where the error was raised.
        /// \param line   The line number of the code line where the error was
        ///               raised.
        /// \param mode   The parameter \p mode specifies whether the constructed
        ///               hpx::error_code belongs to the error category
        ///               \a hpx_category (if mode is \a plain, this is the
        ///               default) or to the category \a hpx_category_rethrow
        ///               (if mode is \a rethrow).
        ///
        /// \throws nothing
        inline error_code(error e, char const* func, char const* file, 
            long line, throwmode mode = plain);

        /// Construct an object of type error_code.
        ///
        /// \param e      The parameter \p e holds the hpx::error code the new
        ///               exception should encapsulate.
        /// \param msg    The parameter \p msg holds the error message the new
        ///               exception should encapsulate.
        /// \param mode   The parameter \p mode specifies whether the constructed
        ///               hpx::error_code belongs to the error category
        ///               \a hpx_category (if mode is \a plain, this is the
        ///               default) or to the category \a hpx_category_rethrow
        ///               (if mode is \a rethrow).
        ///
        /// \throws std#bad_alloc (if allocation of a copy of
        ///         the passed string fails).
        inline error_code(error e, char const* msg, throwmode mode = plain);

        /// Construct an object of type error_code.
        ///
        /// \param e      The parameter \p e holds the hpx::error code the new
        ///               exception should encapsulate.
        /// \param msg    The parameter \p msg holds the error message the new
        ///               exception should encapsulate.
        /// \param func   The name of the function where the error was raised.
        /// \param file   The file name of the code where the error was raised.
        /// \param line   The line number of the code line where the error was
        ///               raised.
        /// \param mode   The parameter \p mode specifies whether the constructed
        ///               hpx::error_code belongs to the error category
        ///               \a hpx_category (if mode is \a plain, this is the
        ///               default) or to the category \a hpx_category_rethrow
        ///               (if mode is \a rethrow).
        ///
        /// \throws std#bad_alloc (if allocation of a copy of
        ///         the passed string fails).
        inline error_code(error e, char const* msg, char const* func,
                char const* file, long line, throwmode mode = plain);

        /// Construct an object of type error_code.
        ///
        /// \param e      The parameter \p e holds the hpx::error code the new
        ///               exception should encapsulate.
        /// \param msg    The parameter \p msg holds the error message the new
        ///               exception should encapsulate.
        /// \param mode   The parameter \p mode specifies whether the constructed
        ///               hpx::error_code belongs to the error category
        ///               \a hpx_category (if mode is \a plain, this is the
        ///               default) or to the category \a hpx_category_rethrow
        ///               (if mode is \a rethrow).
        ///
        /// \throws std#bad_alloc (if allocation of a copy of
        ///         the passed string fails).
        inline error_code(error e, std::string const& msg, throwmode mode = plain);

        /// Construct an object of type error_code.
        ///
        /// \param e      The parameter \p e holds the hpx::error code the new
        ///               exception should encapsulate.
        /// \param msg    The parameter \p msg holds the error message the new
        ///               exception should encapsulate.
        /// \param func   The name of the function where the error was raised.
        /// \param file   The file name of the code where the error was raised.
        /// \param line   The line number of the code line where the error was
        ///               raised.
        /// \param mode   The parameter \p mode specifies whether the constructed
        ///               hpx::error_code belongs to the error category
        ///               \a hpx_category (if mode is \a plain, this is the
        ///               default) or to the category \a hpx_category_rethrow
        ///               (if mode is \a rethrow).
        ///
        /// \throws std#bad_alloc (if allocation of a copy of
        ///         the passed string fails).
        inline error_code(error e, std::string const& msg, char const* func,
                char const* file, long line, throwmode mode = plain);

        /// Return a reference to the error message stored in the hpx::error_code.
        ///
        /// \throws nothing
        inline std::string get_message() const;

        /// \brief Clear this error_code object.
        /// The postconditions of invoking this method are
        /// * value() == hpx::success and category() == hpx::get_hpx_category()
        void clear()
        {
            this->boost::system::error_code::assign(success, get_hpx_category());
            exception_ = boost::exception_ptr();
        }

    private:
        friend boost::exception_ptr detail::access_exception(error_code const&);
        friend class exception;

        inline error_code(int err, hpx::exception const& e);
        boost::exception_ptr exception_;
    };

    /// \brief Predefined error_code object used as "throw on error" tag.
    ///
    /// The predefined hpx::error_code object \a hpx::throws is supplied for use as
    /// a "throw on error" tag.
    ///
    /// Functions that specify an argument in the form 'error_code& ec=throws'
    /// (with appropriate namespace qualifiers), have the following error
    /// handling semantics:
    ///
    /// If &ec != &throws and an error occurred: ec.value() returns the
    /// implementation specific error number for the particular error that
    /// occurred and ec.category() returns the error_category for ec.value().
    ///
    /// If &ec != &throws and an error did not occur, ec.clear().
    ///
    /// If an error occurs and &ec == &throws, the function throws an exception
    /// of type \a hpx::exception or of a type derived from it. The exception's
    /// \a get_errorcode() member function returns a reference to an
    /// \a hpx::error_code object with the behavior as specified above.
    ///
    HPX_EXCEPTION_EXPORT extern error_code throws;

    /// @{
    /// \brief Returns error_code(e, "", mode).
    inline error_code
    make_error_code(error e, throwmode mode = plain)
    {
        return error_code(e, mode);
    }
    inline error_code
    make_error_code(error e, char const* func, char const* file, long line, 
        throwmode mode = plain)
    {
        return error_code(e, func, file, line, mode);
    }

    /// \brief Returns error_code(e, msg, mode).
    inline error_code
    make_error_code(error e, char const* msg, throwmode mode = plain)
    {
        return error_code(e, msg, mode);
    }
    inline error_code
    make_error_code(error e, char const* msg, char const* func,
        char const* file, long line, throwmode mode = plain)
    {
        return error_code(e, msg, func, file, line, mode);
    }

    /// \brief Returns error_code(e, msg, mode).
    inline error_code
    make_error_code(error e, std::string const& msg, throwmode mode = plain)
    {
        return error_code(e, msg, mode);
    }
    inline error_code
    make_error_code(error e, std::string const& msg, char const* func,
        char const* file, long line, throwmode mode = plain)
    {
        return error_code(e, msg, func, file, line, mode);
    }
    ///@}

    /// \brief Returns error_code(hpx::success, "success", mode).
    inline error_code
    make_success_code(throwmode mode = plain)
    {
        return error_code(mode);
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \brief A hpx::exception is the main exception type used by HPX to
    ///        report errors.
    ///
    /// The hpx::exception type is the main exception type  used by HPX to
    /// report errors. Any exceptions thrown by functions in the HPX library
    /// are either of this type or of a type derived from it. This implies that
    /// it is always safe to use this type only in catch statements guarding
    /// HPX library calls.
    class HPX_EXCEPTION_EXPORT exception : public boost::system::system_error
    {
    public:
        /// Construct a hpx::exception from a \a hpx::error.
        ///
        /// \param e    The parameter \p e holds the hpx::error code the new
        ///             exception should encapsulate.
        explicit exception(error e)
          : boost::system::system_error(make_error_code(e, plain))
        {
            BOOST_ASSERT(e >= success && e < last_error);
            LERR_(error) << "created exception: " << this->what();
        }

        /// Construct a hpx::exception from a boost#system_error.
        explicit exception(boost::system::system_error const& e)
          : boost::system::system_error(e)
        {
            LERR_(error) << "created exception: " << this->what();
        }

        /// Construct a hpx::exception from a \a hpx::error and an error message.
        ///
        /// \param e      The parameter \p e holds the hpx::error code the new
        ///               exception should encapsulate.
        /// \param msg    The parameter \p msg holds the error message the new
        ///               exception should encapsulate.
        /// \param mode   The parameter \p mode specifies whether the returned
        ///               hpx::error_code belongs to the error category
        ///               \a hpx_category (if mode is \a plain, this is the
        ///               default) or to the category \a hpx_category_rethrow
        ///               (if mode is \a rethrow).
        exception(error e, char const* msg, throwmode mode = plain)
          : boost::system::system_error(make_system_error_code(e, mode), msg)
        {
            BOOST_ASSERT(e >= success && e < last_error);
            LERR_(error) << "created exception: " << this->what();
        }

        /// Construct a hpx::exception from a \a hpx::error and an error message.
        ///
        /// \param e      The parameter \p e holds the hpx::error code the new
        ///               exception should encapsulate.
        /// \param msg    The parameter \p msg holds the error message the new
        ///               exception should encapsulate.
        /// \param mode   The parameter \p mode specifies whether the returned
        ///               hpx::error_code belongs to the error category
        ///               \a hpx_category (if mode is \a plain, this is the
        ///               default) or to the category \a hpx_category_rethrow
        ///               (if mode is \a rethrow).
        exception(error e, std::string const& msg, throwmode mode = plain)
          : boost::system::system_error(make_system_error_code(e, mode), msg)
        {
            BOOST_ASSERT(e >= success && e < last_error);
            LERR_(error) << "created exception: " << this->what();
        }

        /// Destruct a hpx::exception
        ///
        /// \throws nothing
        ~exception() throw()
        {
        }

        /// The function \a get_error() returns the hpx::error code stored
        /// in the referenced instance of a hpx::exception. It returns
        /// the hpx::error code this exception instance was constructed
        /// from.
        ///
        /// \throws nothing
        error get_error() const throw()
        {
            return static_cast<error>(
                this->boost::system::system_error::code().value());
        }

        /// The function \a get_error_code() returns a hpx::error_code which
        /// represents the same error condition as this hpx::exception instance.
        ///
        /// \param mode   The parameter \p mode specifies whether the returned
        ///               hpx::error_code belongs to the error category
        ///               \a hpx_category (if mode is \a plain, this is the
        ///               default) or to the category \a hpx_category_rethrow
        ///               (if mode is \a rethrow).
        error_code get_error_code(throwmode mode = plain) const throw()
        {
            return error_code(this->boost::system::system_error::code().value(),
                *this);
        }
    };

    /// \cond NODETAIL
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
        struct tag_throw_env {};

        // Stores the information about the locality id the exception has been
        // raised on. This information will show up in error messages under the
        // [locality] tag.
        typedef boost::error_info<detail::tag_throw_locality, boost::uint32_t>
            throw_locality;

        // Stores the information about the hostname of the locality the exception
        // has been raised on. This information will show up in error messages
        // under the [hostname] tag.
        typedef boost::error_info<detail::tag_throw_hostname, std::string>
            throw_hostname;

        // Stores the information about the pid of the OS process the exception
        // has been raised on. This information will show up in error messages
        // under the [pid] tag.
        typedef boost::error_info<detail::tag_throw_pid, boost::int64_t>
            throw_pid;

        // Stores the information about the shepherd thread the exception has been
        // raised on. This information will show up in error messages under the
        // [shepherd] tag.
        typedef boost::error_info<detail::tag_throw_shepherd, std::size_t>
            throw_shepherd;

        // Stores the information about the HPX thread the exception has been
        // raised on. This information will show up in error messages under the
        // [thread_id] tag.
        typedef boost::error_info<detail::tag_throw_thread_id, std::size_t>
            throw_thread_id;

        // Stores the information about the HPX thread name the exception has been
        // raised on. This information will show up in error messages under the
        // [thread_name] tag.
        typedef boost::error_info<detail::tag_throw_thread_name, std::string>
            throw_thread_name;

        // Stores the information about the function name the exception has been
        // raised in. This information will show up in error messages under the
        // [function] tag.
        typedef boost::error_info<detail::tag_throw_function, std::string>
            throw_function;

        // Stores the information about the source file name the exception has
        // been raised in. This information will show up in error messages under
        // the [file] tag.
        typedef boost::error_info<detail::tag_throw_file, std::string>
            throw_file;

        // Stores the information about the source file line number the exception
        // has been raised at. This information will show up in error messages
        // under the [line] tag.
        using boost::throw_line;

        // Stores the information about the stack backtrace at the point the
        // exception has been raised at. This information will show up in error
        // messages under the [stack_trace] tag.
        typedef boost::error_info<detail::tag_throw_stacktrace, std::string>
            throw_stacktrace;

        // Stores the full execution environment of the locality the exception
        // has been raised in. This information will show up in error messages
        // under the [env] tag.
        typedef boost::error_info<detail::tag_throw_env, std::string>
            throw_env;

        // construct an exception, internal helper
        template <typename Exception>
        HPX_EXPORT boost::exception_ptr
            construct_exception(Exception const& e,
                std::string const& func, std::string const& file, long line,
                std::string const& back_trace, boost::uint32_t node = 0,
                std::string const& hostname = "", boost::int64_t pid = -1,
                std::size_t shepherd = ~0, std::size_t thread_id = 0,
                std::string const& thread_name = "",
                std::string const& env = "");

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

        // Report an early or late exception and locally abort execution. There
        // isn't anything more we could do.
        HPX_EXPORT void report_exception_and_terminate(boost::exception_ptr const&);
    }
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Extract the diagnostic information embedded in the given
    /// exception and return a string holding a formatted message.
    ///
    /// The function \a hpx::diagnostic_information can be used to extract all
    /// diagnostic information stored in the given exception instance as a
    /// formatted string. This simplifies debug output as it composes the
    /// diagnostics into one, easy to use function call. This includes
    /// the name of the source file and line number, the sequence number of the
    /// OS-thread and the HPX-thread id, the locality id and the stack backtrace
    /// of the point where the original exception was thrown.
    ///
    /// \param e    The parameter \p e will be inspected for all diagnostic
    ///             information elements which have been stored at the point
    ///             where the exception was thrown. This parameter can be one
    ///             of the following types: \a hpx::exception, \a hox::error_code,
    ///             \a boost::exception, or \a boost::exception_ptr
    ///
    /// \returns    The formatted string holding all of the available
    ///             diagnostic information stored in the given exception
    ///             instance.
    ///
    /// \throws     std#bad_alloc (if any of the required allocation operations
    ///             fail)
    ///
    /// \see        \a hpx::get_locality_id(), \a hpx::get_host_name(),
    ///             \a hpx::get_process_id(), \a hpx::get_function_name(),
    ///             \a hpx::get_file_name(), \a hpx::get_line_number(),
    ///             \a hpx::get_os_thread(), \a hpx::get_thread_id(),
    ///             \a hpx::get_thread_description()
    ///
    HPX_EXPORT std::string diagnostic_information(hpx::exception const& e);

    /// \copydoc diagnostic_information(hpx::exception const& e)
    HPX_EXPORT std::string diagnostic_information(hpx::error_code const& e);

    /// \cond NOINTERNAL
    HPX_EXPORT std::string diagnostic_information(boost::exception const& e);
    HPX_EXPORT std::string diagnostic_information(boost::exception_ptr const& e);
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    // Extract elements of the diagnostic information embedded in the given
    // exception.

    /// \brief Return the locality id where the exception was thrown.
    ///
    /// The function \a hpx::get_locality_id can be used to extract the
    /// diagnostic information element representing the locality id as stored
    /// in the given exception instance.
    ///
    /// \param e    The parameter \p e will be inspected for all diagnostic
    ///             information elements which have been stored at the point
    ///             where the exception was thrown. This parameter can be one
    ///             of the following types: \a hpx::exception, \a hox::error_code,
    ///             \a boost::exception, or \a boost::exception_ptr
    ///
    /// \returns    The locality id of the locality where the exception was
    ///             thrown. If the exception instance does not hold
    ///             this information, the function will return
    ///             \a hpx::naming#invalid_locality_id.
    ///
    /// \throws     nothing
    ///
    /// \see        \a hpx::diagnostic_information(), \a hpx::get_host_name(),
    ///             \a hpx::get_process_id(), \a hpx::get_function_name(),
    ///             \a hpx::get_file_name(), \a hpx::get_line_number(),
    ///             \a hpx::get_os_thread(), \a hpx::get_thread_id(),
    ///             \a hpx::get_thread_description()
    ///
    HPX_EXPORT boost::uint32_t get_locality_id(hpx::exception const& e);

    /// \copydoc get_locality_id(hpx::exception const& e)
    HPX_EXPORT boost::uint32_t get_locality_id(hpx::error_code const& e);

    /// \cond NOINTERNAL
    HPX_EXPORT boost::uint32_t get_locality_id(boost::exception const& e);
    HPX_EXPORT boost::uint32_t get_locality_id(boost::exception_ptr const& e);
    /// \endcond

    /// \brief Return the hostname of the locality where the exception was
    ///        thrown.
    ///
    /// The function \a hpx::get_host_name can be used to extract the
    /// diagnostic information element representing the host name as stored in
    /// the given exception instance.
    ///
    /// \param e    The parameter \p e will be inspected for all diagnostic
    ///             information elements which have been stored at the point
    ///             where the exception was thrown. This parameter can be one
    ///             of the following types: \a hpx::exception, \a hox::error_code,
    ///             \a boost::exception, or \a boost::exception_ptr
    ///
    /// \returns    The hostname of the locality where the exception was
    ///             thrown. If the exception instance does not hold
    ///             this information, the function will return and empty string.
    ///
    /// \throws     std#bad_alloc (if one of the required allocations fails)
    ///
    HPX_EXPORT std::string get_host_name(hpx::exception const& e);

    /// \copydoc get_host_name(hpx::exception const& e)
    HPX_EXPORT std::string get_host_name(hpx::error_code const& e);

    /// \cond NOINTERNAL
    HPX_EXPORT std::string get_host_name(boost::exception const& e);
    HPX_EXPORT std::string get_host_name(boost::exception_ptr const& e);
    /// \endcond

    /// \brief Return the (operating system) process id of the locality where
    ///        the exception was thrown.
    ///
    /// The function \a hpx::get_process_id can be used to extract the
    /// diagnostic information element representing the process id as stored in
    /// the given exception instance.
    ///
    /// \returns    The process id of the OS-process which threw the exception
    ///             If the exception instance does not hold
    ///             this information, the function will return 0.
    ///
    /// \param e    The parameter \p e will be inspected for all diagnostic
    ///             information elements which have been stored at the point
    ///             where the exception was thrown. This parameter can be one
    ///             of the following types: \a hpx::exception, \a hox::error_code,
    ///             \a boost::exception, or \a boost::exception_ptr
    ///
    /// \throws     nothing
    ///
    HPX_EXPORT boost::int64_t get_process_id(hpx::exception const& e);

    /// \copydoc get_process_id(hpx::exception const& e)
    HPX_EXPORT boost::int64_t get_process_id(hpx::error_code const& e);

    /// \cond NOINTERNAL
    HPX_EXPORT boost::int64_t get_process_id(boost::exception const& e);
    HPX_EXPORT boost::int64_t get_process_id(boost::exception_ptr const& e);
    /// \endcond

    /// \brief Return the function name from which the exception was thrown.
    ///
    /// The function \a hpx::get_function_name can be used to extract the
    /// diagnostic information element representing the name of the function
    /// as stored in the given exception instance.
    ///
    /// \returns    The name of the function from which the exception was
    ///             thrown. If the exception instance does not hold
    ///             this information, the function will return an empty string.
    ///
    /// \param e    The parameter \p e will be inspected for all diagnostic
    ///             information elements which have been stored at the point
    ///             where the exception was thrown. This parameter can be one
    ///             of the following types: \a hpx::exception, \a hox::error_code,
    ///             \a boost::exception, or \a boost::exception_ptr
    ///
    /// \throws     std#bad_alloc (if one of the required allocations fails)
    ///
    HPX_EXPORT std::string get_function_name(hpx::exception const& e);

    /// \copydoc get_function_name(hpx::exception const& e)
    HPX_EXPORT std::string get_function_name(hpx::error_code const& e);

    /// \cond NOINTERNAL
    HPX_EXPORT std::string get_function_name(boost::exception const& e);
    HPX_EXPORT std::string get_function_name(boost::exception_ptr const& e);
    /// \endcond

    /// \brief Return the (source code) file name of the function from which
    ///        the exception was thrown.
    ///
    /// The function \a hpx::get_file_name can be used to extract the
    /// diagnostic information element representing the name of the source file
    /// as stored in the given exception instance.
    ///
    /// \returns    The name of the source file of the function from which the
    ///             exception was thrown. If the exception instance does
    ///             not hold this information, the function will return an empty
    ///             string.
    ///
    /// \param e    The parameter \p e will be inspected for all diagnostic
    ///             information elements which have been stored at the point
    ///             where the exception was thrown. This parameter can be one
    ///             of the following types: \a hpx::exception, \a hox::error_code,
    ///             \a boost::exception, or \a boost::exception_ptr
    ///
    /// \throws     std#bad_alloc (if one of the required allocations fails)
    ///
    HPX_EXPORT std::string get_file_name(hpx::exception const& e);

    /// \copydoc get_file_name(hpx::exception const& e)
    HPX_EXPORT std::string get_file_name(hpx::error_code const& e);

    /// \cond NOINTERNAL
    HPX_EXPORT std::string get_file_name(boost::exception const& e);
    HPX_EXPORT std::string get_file_name(boost::exception_ptr const& e);
    /// \endcond

    /// \brief Return the line number in the (source code) file of the function
    ///        from which the exception was thrown.
    ///
    /// The function \a hpx::get_line_number can be used to extract the
    /// diagnostic information element representing the line number
    /// as stored in the given exception instance.
    ///
    /// \returns    The line number of the place where the exception was
    ///             thrown. If the exception instance does not hold
    ///             this information, the function will return -1.
    ///
    /// \param e    The parameter \p e will be inspected for all diagnostic
    ///             information elements which have been stored at the point
    ///             where the exception was thrown. This parameter can be one
    ///             of the following types: \a hpx::exception, \a hox::error_code,
    ///             \a boost::exception, or \a boost::exception_ptr
    ///
    /// \throws     nothing
    ///
    HPX_EXPORT int get_line_number(hpx::exception const& e);

    /// \copydoc get_line_number(hpx::exception const& e)
    HPX_EXPORT int get_line_number(hpx::error_code const& e);

    /// \cond NOINTERNAL
    HPX_EXPORT int get_line_number(boost::exception const& e);
    HPX_EXPORT int get_line_number(boost::exception_ptr const& e);
    /// \endcond

    /// \brief Return the sequence number of the OS-thread used to execute
    ///        HPX-threads from which the exception was thrown.
    ///
    /// The function \a hpx::get_os_thread can be used to extract the
    /// diagnostic information element representing the sequence number  of the
    /// OS-thread as stored in the given exception instance.
    ///
    /// \returns    The sequence number of the OS-thread used to execute the
    ///             HPX-thread from which the exception was
    ///             thrown. If the exception instance does not hold
    ///             this information, the function will return std::size(-1).
    ///
    /// \param e    The parameter \p e will be inspected for all diagnostic
    ///             information elements which have been stored at the point
    ///             where the exception was thrown. This parameter can be one
    ///             of the following types: \a hpx::exception, \a hox::error_code,
    ///             \a boost::exception, or \a boost::exception_ptr
    ///
    /// \throws     nothing
    ///
    HPX_EXPORT std::size_t get_os_thread(hpx::exception const& e);

    /// \copydoc get_os_thread(hpx::exception const& e)
    HPX_EXPORT std::size_t get_os_thread(hpx::error_code const& e);

    /// \cond NOINTERNAL
    HPX_EXPORT std::size_t get_os_thread(boost::exception const& e);
    HPX_EXPORT std::size_t get_os_thread(boost::exception_ptr const& e);
    /// \endcond

    /// \brief Return the unique thread id of the HPX-thread from which the
    ///        exception was thrown.
    ///
    /// The function \a hpx::get_thread_id can be used to extract the
    /// diagnostic information element representing the HPX-thread id
    /// as stored in the given exception instance.
    ///
    /// \returns    The unique thread id of the HPX-thread from which the
    ///             exception was thrown. If the exception instance
    ///             does not hold this information, the function will return
    ///             std::size_t(0).
    ///
    /// \param e    The parameter \p e will be inspected for all diagnostic
    ///             information elements which have been stored at the point
    ///             where the exception was thrown. This parameter can be one
    ///             of the following types: \a hpx::exception, \a hox::error_code,
    ///             \a boost::exception, or \a boost::exception_ptr
    ///
    /// \throws     nothing
    ///
    HPX_EXPORT std::size_t get_thread_id(hpx::exception const& e);

    /// \copydoc get_thread_id(hpx::exception const& e)
    HPX_EXPORT std::size_t get_thread_id(hpx::error_code const& e);

    /// \cond NOINTERNAL
    HPX_EXPORT std::size_t get_thread_id(boost::exception const& e);
    HPX_EXPORT std::size_t get_thread_id(boost::exception_ptr const& e);
    /// \endcond

    /// \brief Return any additionally available thread description of the
    ///        HPX-thread from which the exception was thrown.
    ///
    /// The function \a hpx::get_thread_description can be used to extract the
    /// diagnostic information element representing the additional thread
    /// description as stored in the given exception instance.
    ///
    /// \returns    Any additionally available thread description of the
    ///             HPX-thread from which the exception was
    ///             thrown. If the exception instance does not hold
    ///             this information, the function will return an empty string.
    ///
    /// \param e    The parameter \p e will be inspected for all diagnostic
    ///             information elements which have been stored at the point
    ///             where the exception was thrown. This parameter can be one
    ///             of the following types: \a hpx::exception, \a hox::error_code,
    ///             \a boost::exception, or \a boost::exception_ptr
    ///
    /// \throws     std#bad_alloc (if one of the required allocations fails)
    ///
    HPX_EXPORT std::string get_thread_description(hpx::exception const& e);

    /// \copydoc get_thread_description(hpx::exception const& e)
    HPX_EXPORT std::string get_thread_description(hpx::error_code const& e);

    /// \cond NOINTERNAL
    HPX_EXPORT std::string get_thread_description(boost::exception const& e);
    HPX_EXPORT std::string get_thread_description(boost::exception_ptr const& e);
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    // \cond NOINTERNAL
    inline error_code::error_code(error e, throwmode mode)
      : boost::system::error_code(make_system_error_code(e, mode))
    {
        if (e != success && e != no_success && !(mode & lightweight))
            exception_ = detail::get_exception(hpx::exception(e, "", mode));
    }

    inline error_code::error_code(error e, char const* func,
            char const* file, long line, throwmode mode)
      : boost::system::error_code(make_system_error_code(e, mode))
    {
        if (e != success && e != no_success && !(mode & lightweight)) {
            boost::filesystem::path p(hpx::util::create_path(file));
            exception_ = detail::get_exception(hpx::exception(e, "", mode),
                func, p.string(), line);
        }
    }

    inline error_code::error_code(error e, char const* msg, throwmode mode)
      : boost::system::error_code(make_system_error_code(e, mode))
    {
        if (e != success && e != no_success && !(mode & lightweight))
            exception_ = detail::get_exception(hpx::exception(e, msg, mode));
    }

    inline error_code::error_code(error e, char const* msg,
            char const* func, char const* file, long line, throwmode mode)
      : boost::system::error_code(make_system_error_code(e, mode))
    {
        if (e != success && e != no_success && !(mode & lightweight)) {
            boost::filesystem::path p(hpx::util::create_path(file));
            exception_ = detail::get_exception(hpx::exception(e, msg, mode),
                func, p.string(), line);
        }
    }

    inline error_code::error_code(error e, std::string const& msg,
            throwmode mode)
      : boost::system::error_code(make_system_error_code(e, mode))
    {
        if (e != success && e != no_success && !(mode & lightweight))
            exception_ = detail::get_exception(hpx::exception(e, msg, mode));
    }

    inline error_code::error_code(error e, std::string const& msg,
            char const* func, char const* file, long line, throwmode mode)
      : boost::system::error_code(make_system_error_code(e, mode))
    {
        if (e != success && e != no_success && !(mode & lightweight)) {
            boost::filesystem::path p(hpx::util::create_path(file));
            exception_ = detail::get_exception(hpx::exception(e, msg, mode),
                func, p.string(), line);
        }
    }

    inline error_code::error_code(int err, hpx::exception const& e)
    {
        this->boost::system::error_code::assign(err, get_hpx_category());
        try {
            throw e;
        }
        catch (...) {
            exception_ = boost::current_exception();
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    inline std::string error_code::get_message() const
    {
        if (exception_) {
            try {
                boost::rethrow_exception(exception_);
            }
            catch (boost::exception const& be) {
                return dynamic_cast<std::exception const*>(&be)->what();
            }
        }
        return "";
    }
    // \endcond
}

/// \cond NOEXTERNAL
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
/// \endcond

/// \cond NOINTERNAL
///////////////////////////////////////////////////////////////////////////////
// helper macro allowing to prepend file name and line number to a generated
// exception
#define HPX_THROW_EXCEPTION_EX(except, errcode, func, msg, mode)              \
    {                                                                         \
        boost::filesystem::path p__(hpx::util::create_path(__FILE__));        \
        hpx::detail::throw_exception(                                         \
            except(static_cast<hpx::error>(errcode), msg, mode),              \
            func, p__.string(), __LINE__);                                    \
    }                                                                         \
    /**/

#define HPX_THROW_STD_EXCEPTION(except, func)                                 \
    {                                                                         \
        boost::filesystem::path p__(hpx::util::create_path(__FILE__));        \
        hpx::detail::throw_exception(except, func, p__.string(), __LINE__);   \
    }                                                                         \
    /**/

#define HPX_RETHROW_EXCEPTION(errcode, f, msg)                                \
    HPX_THROW_EXCEPTION_EX(hpx::exception, errcode, f, msg, hpx::rethrow)     \
    /**/

#define HPX_RETHROWS_IF(ec, errcode, f, msg)                                  \
    {                                                                         \
        if (&ec == &hpx::throws) {                                            \
            HPX_RETHROW_EXCEPTION(errcode, f, msg);                           \
        } else {                                                              \
            ec = make_error_code(static_cast<hpx::error>(errcode), msg, f,    \
                __FILE__, __LINE__,                                           \
                ec.category() == hpx::get_lightweight_hpx_category()) ?       \
                    hpx::lightweight_rethrow : hpx::rethrow);                 \
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
            ec = make_error_code(static_cast<hpx::error>(errcode), msg,       \
                BOOST_CURRENT_FUNCTION, __FILE__, __LINE__,                   \
                (ec.category() == hpx::get_lightweight_hpx_category()) ?      \
                    hpx::lightweight : hpx::plain);                           \
        }                                                                     \
    }                                                                         \
    /**/

#define HPX_RETHROWS_IN_CURRENT_FUNC_IF(ec, errcode, msg)                     \
    {                                                                         \
        if (&ec == &hpx::throws) {                                            \
            HPX_RETHROW_EXCEPTION(errcode, BOOST_CURRENT_FUNCTION, msg);      \
        } else {                                                              \
            ec = make_error_code(static_cast<hpx::error>(errcode), msg,       \
                BOOST_CURRENT_FUNCTION, __FILE__, __LINE__,                   \
                (ec.category() == hpx::get_lightweight_hpx_category()) ?      \
                    (hpx::lightweight_rethrow) : hpx::rethrow);               \
        }                                                                     \
    }                                                                         \
    /**/
/// \endcond

///////////////////////////////////////////////////////////////////////////////
/// \def HPX_THROW_EXCEPTION(errcode, f, msg)
/// \brief Throw a hpx::exception initialized from the given parameters
///
/// The macro \a HPX_THROW_EXCEPTION can be used to throw a hpx::exception.
/// The purpose of this macro is to prepend the source file name and line number
/// of the position where the exception is thrown to the error message.
/// Moreover, this associates additional diagnostic information with the
/// exception, such as file name and line number, locality id and thread id,
/// and stack backtrace from the point where the exception was thrown.
///
/// The parameter \p errcode holds the hpx::error code the new exception should
/// encapsulate. The parameter \p f is expected to hold the name of the
/// function exception is thrown from and the parameter \p msg holds the error
/// message the new exception should encapsulate.
///
/// \par Example:
///
/// \code
///      void raise_exception()
///      {
///          // Throw a hpx::exception initialized from the given parameters.
///          // Additionally associate with this exception some detailed
///          // diagnostic information about the throw-site.
///          HPX_THROW_EXCEPTION(hpx::no_success, "raise_exception", "simulated error");
///      }
/// \endcode
///
#define HPX_THROW_EXCEPTION(errcode, f, msg)                                  \
    HPX_THROW_EXCEPTION_EX(hpx::exception, errcode, f, msg, hpx::plain)       \
    /**/

/// \def HPX_THROWS_IF(ec, errcode, f, msg)
/// \brief Either throw a hpx::exception or initialize \a hpx::error_code from
///        the given parameters
///
/// The macro \a HPX_THROWS_IF can be used to either throw a \a hpx::exception
/// or to initialize a \a hpx::error_code from the given parameters. If
/// &ec == &hpx::throws, the semantics of this macro are equivalent to
/// \a HPX_THROW_EXCEPTION. If &ec != &hpx::throws, the \a hpx::error_code
/// instance \p ec is initialized instead.
///
/// The parameter \p errcode holds the hpx::error code from which the new
/// exception should be initialized. The parameter \p f is expected to hold the
/// name of the function exception is thrown from and the parameter \p msg
/// holds the error message the new exception should encapsulate.
///
#define HPX_THROWS_IF(ec, errcode, f, msg)                                    \
    {                                                                         \
        if (&ec == &hpx::throws) {                                            \
            HPX_THROW_EXCEPTION(errcode, f, msg);                             \
        } else {                                                              \
            ec = make_error_code(static_cast<hpx::error>(errcode), msg, f,    \
                __FILE__, __LINE__,                                           \
                (ec.category() == hpx::get_lightweight_hpx_category()) ?      \
                    hpx::lightweight : hpx::plain);                           \
        }                                                                     \
    }                                                                         \
    /**/

#include <hpx/config/warnings_suffix.hpp>

#endif

