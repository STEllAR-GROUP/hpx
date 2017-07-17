//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file exception.hpp

#if !defined(HPX_EXCEPTION_MAR_24_2008_0929AM)
#define HPX_EXCEPTION_MAR_24_2008_0929AM

#include <hpx/config.hpp>
#include <hpx/error.hpp>
#include <hpx/error_code.hpp>
#include <hpx/exception_fwd.hpp>
#include <hpx/exception_info.hpp>
#include <hpx/runtime/naming_fwd.hpp>

#include <boost/system/error_code.hpp>
#include <boost/system/system_error.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
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
        explicit exception(error e = success);

        /// Construct a hpx::exception from a boost#system_error.
        explicit exception(boost::system::system_error const& e);

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
        exception(error e, char const* msg, throwmode mode = plain);

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
        exception(error e, std::string const& msg, throwmode mode = plain);

        /// Destruct a hpx::exception
        ///
        /// \throws nothing
        ~exception() throw();

        /// The function \a get_error() returns the hpx::error code stored
        /// in the referenced instance of a hpx::exception. It returns
        /// the hpx::error code this exception instance was constructed
        /// from.
        ///
        /// \throws nothing
        error get_error() const noexcept;

        /// The function \a get_error_code() returns a hpx::error_code which
        /// represents the same error condition as this hpx::exception instance.
        ///
        /// \param mode   The parameter \p mode specifies whether the returned
        ///               hpx::error_code belongs to the error category
        ///               \a hpx_category (if mode is \a plain, this is the
        ///               default) or to the category \a hpx_category_rethrow
        ///               (if mode is \a rethrow).
        error_code get_error_code(throwmode mode = plain) const noexcept;
    };

    ///////////////////////////////////////////////////////////////////////////
    /// \brief A hpx::thread_interrupted is the exception type used by HPX to
    ///        interrupt a running HPX thread.
    ///
    /// The \a hpx::thread_interrupted type is the exception type used by HPX to
    /// interrupt a running thread.
    ///
    /// A running thread can be interrupted by invoking the interrupt() member
    /// function of the corresponding hpx::thread object. When the interrupted
    /// thread next executes one of the specified interruption points (or if it
    /// is currently blocked whilst executing one) with interruption enabled,
    /// then a hpx::thread_interrupted exception will be thrown in the interrupted
    /// thread. If not caught, this will cause the execution of the interrupted
    /// thread to terminate. As with any other exception, the stack will be
    /// unwound, and destructors for objects of automatic storage duration will
    /// be executed.
    ///
    /// If a thread wishes to avoid being interrupted, it can create an instance
    /// of \a hpx::this_thread::disable_interruption. Objects of this class disable
    /// interruption for the thread that created them on construction, and
    /// restore the interruption state to whatever it was before on destruction.
    ///
    /// \code
    ///     void f()
    ///     {
    ///         // interruption enabled here
    ///         {
    ///             hpx::this_thread::disable_interruption di;
    ///             // interruption disabled
    ///             {
    ///                 hpx::this_thread::disable_interruption di2;
    ///                 // interruption still disabled
    ///             } // di2 destroyed, interruption state restored
    ///             // interruption still disabled
    ///         } // di destroyed, interruption state restored
    ///         // interruption now enabled
    ///     }
    /// \endcode
    ///
    /// The effects of an instance of \a hpx::this_thread::disable_interruption can be
    /// temporarily reversed by constructing an instance of
    /// \a hpx::this_thread::restore_interruption, passing in the
    /// \a hpx::this_thread::disable_interruption object in question. This will restore
    /// the interruption state to what it was when the
    /// \a hpx::this_thread::disable_interruption
    /// object was constructed, and then disable interruption again when the
    /// \a hpx::this_thread::restore_interruption object is destroyed.
    ///
    /// \code
    ///     void g()
    ///     {
    ///         // interruption enabled here
    ///         {
    ///             hpx::this_thread::disable_interruption di;
    ///             // interruption disabled
    ///             {
    ///                 hpx::this_thread::restore_interruption ri(di);
    ///                 // interruption now enabled
    ///             } // ri destroyed, interruption disable again
    ///         } // di destroyed, interruption state restored
    ///         // interruption now enabled
    ///     }
    /// \endcode
    ///
    /// At any point, the interruption state for the current thread can be
    /// queried by calling \a hpx::this_thread::interruption_enabled().
    struct HPX_EXCEPTION_EXPORT thread_interrupted : std::exception {};

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

        ///////////////////////////////////////////////////////////////////////
        // Stores the information about the locality id the exception has been
        // raised on. This information will show up in error messages under the
        // [locality] tag.
        HPX_DEFINE_ERROR_INFO(throw_locality, std::uint32_t);

        // Stores the information about the hostname of the locality the exception
        // has been raised on. This information will show up in error messages
        // under the [hostname] tag.
        HPX_DEFINE_ERROR_INFO(throw_hostname, std::string);

        // Stores the information about the pid of the OS process the exception
        // has been raised on. This information will show up in error messages
        // under the [pid] tag.
        HPX_DEFINE_ERROR_INFO(throw_pid, std::int64_t);

        // Stores the information about the shepherd thread the exception has been
        // raised on. This information will show up in error messages under the
        // [shepherd] tag.
        HPX_DEFINE_ERROR_INFO(throw_shepherd, std::size_t);

        // Stores the information about the HPX thread the exception has been
        // raised on. This information will show up in error messages under the
        // [thread_id] tag.
        HPX_DEFINE_ERROR_INFO(throw_thread_id, std::size_t);

        // Stores the information about the HPX thread name the exception has been
        // raised on. This information will show up in error messages under the
        // [thread_name] tag.
        HPX_DEFINE_ERROR_INFO(throw_thread_name, std::string);

        // Stores the information about the function name the exception has been
        // raised in. This information will show up in error messages under the
        // [function] tag.
        HPX_DEFINE_ERROR_INFO(throw_function, std::string);

        // Stores the information about the source file name the exception has
        // been raised in. This information will show up in error messages under
        // the [file] tag.
        HPX_DEFINE_ERROR_INFO(throw_file, std::string);

        // Stores the information about the source file line number the exception
        // has been raised at. This information will show up in error messages
        // under the [line] tag.
        HPX_DEFINE_ERROR_INFO(throw_line, long);

        // Stores the information about the stack backtrace at the point the
        // exception has been raised at. This information will show up in error
        // messages under the [stack_trace] tag.
        HPX_DEFINE_ERROR_INFO(throw_stacktrace, std::string);

        // Stores the full execution environment of the locality the exception
        // has been raised in. This information will show up in error messages
        // under the [env] tag.
        HPX_DEFINE_ERROR_INFO(throw_env, std::string);

        // Stores the full HPX configuration information of the locality the
        // exception has been raised in. This information will show up in error
        // messages under the [config] tag.
        HPX_DEFINE_ERROR_INFO(throw_config, std::string);

        // Stores the current runtime state. This information will show up in
        // error messages under the [state] tag.
        HPX_DEFINE_ERROR_INFO(throw_state, std::string);

        // Stores additional auxiliary information (such as information about
        // the current parcel). This information will show up in error messages
        // under the [auxinfo] tag.
        HPX_DEFINE_ERROR_INFO(throw_auxinfo, std::string);

        // construct an exception, internal helper
        template <typename Exception>
        HPX_EXPORT std::exception_ptr
            construct_exception(Exception const& e,
                std::string const& func, std::string const& file, long line,
                std::string const& back_trace = "", std::uint32_t node = 0,
                std::string const& hostname = "", std::int64_t pid = -1,
                std::size_t shepherd = ~0, std::size_t thread_id = 0,
                std::string const& thread_name = "",
                std::string const& env = "", std::string const& config = "",
                std::string const& state = "", std::string const& auxinfo = "");

        template <typename Exception>
        HPX_EXPORT std::exception_ptr
            construct_lightweight_exception(Exception const& e);

        // HPX_ASSERT handler
        HPX_ATTRIBUTE_NORETURN HPX_EXPORT
        void assertion_failed(char const* expr, char const* function,
            char const* file, long line);

        // HPX_ASSERT_MSG handler
        HPX_ATTRIBUTE_NORETURN HPX_EXPORT
        void assertion_failed_msg(char const* msg, char const* expr,
            char const* function, char const* file, long line);

        // If backtrace support is enabled, this function returns the current
        // stack backtrace, otherwise it will return an empty string.
        HPX_EXPORT std::string backtrace(
            std::size_t frames = HPX_HAVE_THREAD_BACKTRACE_DEPTH);
        HPX_EXPORT std::string backtrace_direct(
            std::size_t frames = HPX_HAVE_THREAD_BACKTRACE_DEPTH);

        // Portably extract the current execution environment
        HPX_EXPORT std::string get_execution_environment();

        // Report an early or late exception and locally abort execution. There
        // isn't anything more we could do.
        HPX_EXPORT void report_exception_and_terminate(std::exception_ptr const&);
        HPX_EXPORT void report_exception_and_terminate(hpx::exception const&);

        // Report an early or late exception and locally exit execution. There
        // isn't anything more we could do. The exception will be re-thrown
        // from hpx::init
        HPX_EXPORT void report_exception_and_continue(std::exception_ptr const&);
        HPX_EXPORT void report_exception_and_continue(hpx::exception const&);
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
    ///             of the following types: \a hpx::exception_info,
    ///             \a hpx::error_code, \a std::exception, or
    ///             \a std::exception_ptr.
    ///
    /// \returns    The formatted string holding all of the available
    ///             diagnostic information stored in the given exception
    ///             instance.
    ///
    /// \throws     std#bad_alloc (if any of the required allocation operations
    ///             fail)
    ///
    /// \see        \a hpx::get_error_locality_id(), \a hpx::get_error_host_name(),
    ///             \a hpx::get_error_process_id(), \a hpx::get_error_function_name(),
    ///             \a hpx::get_error_file_name(), \a hpx::get_error_line_number(),
    ///             \a hpx::get_error_os_thread(), \a hpx::get_error_thread_id(),
    ///             \a hpx::get_error_thread_description(), \a hpx::get_error(),
    ///             \a hpx::get_error_backtrace(), \a hpx::get_error_env(),
    ///             \a hpx::get_error_what(), \a hpx::get_error_config(),
    ///             \a hpx::get_error_state()
    ///
    HPX_EXPORT std::string diagnostic_information(exception_info const& xi);

    /// \cond NOINTERNAL
    template <typename E>
    std::string diagnostic_information(E const& e)
    {
        if (exception_info const* xi = get_exception_info(e))
            return diagnostic_information(*xi);
        return std::string("<unknown>");
    }
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    // Extract elements of the diagnostic information embedded in the given
    // exception.

    /// \brief Return the error message of the thrown exception.
    ///
    /// The function \a hpx::get_error_what can be used to extract the
    /// diagnostic information element representing the error message as stored
    /// in the given exception instance.
    ///
    /// \param e    The parameter \p e will be inspected for the requested
    ///             diagnostic information elements which have been stored at
    ///             the point where the exception was thrown. This parameter
    ///             can be one of the following types: \a hpx::exception_info,
    ///             \a hpx::error_code, \a std::exception, or
    ///             \a std::exception_ptr.
    ///
    /// \returns    The error message stored in the exception
    ///             If the exception instance does not hold
    ///             this information, the function will return an empty string.
    ///
    /// \throws     std#bad_alloc (if one of the required allocations fails)
    ///
    /// \see        \a hpx::diagnostic_information(), \a hpx::get_error_host_name(),
    ///             \a hpx::get_error_process_id(), \a hpx::get_error_function_name(),
    ///             \a hpx::get_error_file_name(), \a hpx::get_error_line_number(),
    ///             \a hpx::get_error_os_thread(), \a hpx::get_error_thread_id(),
    ///             \a hpx::get_error_thread_description(), \a hpx::get_error()
    ///             \a hpx::get_error_backtrace(), \a hpx::get_error_env(),
    ///             \a hpx::get_error_config(), \a hpx::get_error_state()
    ///
    HPX_EXPORT std::string get_error_what(exception_info const& xi);

    /// \cond NOINTERNAL
    template <typename E>
    std::string get_error_what(E const& e)
    {
        if (exception_info const* xi = get_exception_info(e))
            return get_error_what(*xi);
        return std::string("<unknown>");
    }

    inline std::string get_error_what(hpx::error_code const& e)
    {
        // if this is a lightweight error_code, return canned response
        if (e.category() == hpx::get_lightweight_hpx_category())
            return e.message();

        return get_error_what<hpx::error_code>(e);
    }

    inline std::string get_error_what(std::exception const& e)
    {
        return e.what();
    }
    /// \endcond

    /// \brief Return the locality id where the exception was thrown.
    ///
    /// The function \a hpx::get_error_locality_id can be used to extract the
    /// diagnostic information element representing the locality id as stored
    /// in the given exception instance.
    ///
    /// \param e    The parameter \p e will be inspected for the requested
    ///             diagnostic information elements which have been stored at
    ///             the point where the exception was thrown. This parameter
    ///             can be one of the following types: \a hpx::exception_info,
    ///             \a hpx::error_code, \a std::exception, or
    ///             \a std::exception_ptr.
    ///
    /// \returns    The locality id of the locality where the exception was
    ///             thrown. If the exception instance does not hold
    ///             this information, the function will return
    ///             \a hpx::naming#invalid_locality_id.
    ///
    /// \throws     nothing
    ///
    /// \see        \a hpx::diagnostic_information(), \a hpx::get_error_host_name(),
    ///             \a hpx::get_error_process_id(), \a hpx::get_error_function_name(),
    ///             \a hpx::get_error_file_name(), \a hpx::get_error_line_number(),
    ///             \a hpx::get_error_os_thread(), \a hpx::get_error_thread_id(),
    ///             \a hpx::get_error_thread_description(), \a hpx::get_error(),
    ///             \a hpx::get_error_backtrace(), \a hpx::get_error_env(),
    ///             \a hpx::get_error_what(), \a hpx::get_error_config(),
    ///             \a hpx::get_error_state()
    ///
    HPX_EXPORT std::uint32_t get_error_locality_id(hpx::exception_info const& xi);

    /// \cond NOINTERNAL
    template <typename E>
    std::uint32_t get_error_locality_id(E const& e)
    {
        if (exception_info const* xi = get_exception_info(e))
            return get_error_locality_id(*xi);
        return naming::invalid_locality_id;
    }
    /// \endcond

    /// \brief Return the locality id where the exception was thrown.
    ///
    /// The function \a hpx::get_error can be used to extract the
    /// diagnostic information element representing the error value code as
    /// stored in the given exception instance.
    ///
    /// \param e    The parameter \p e will be inspected for the requested
    ///             diagnostic information elements which have been stored at
    ///             the point where the exception was thrown. This parameter
    ///             can be one of the following types: \a hpx::exception,
    ///             \a hpx::error_code, or \a std::exception_ptr.
    ///
    /// \returns    The error value code of the locality where the exception was
    ///             thrown. If the exception instance does not hold
    ///             this information, the function will return
    ///             \a hpx::naming#invalid_locality_id.
    ///
    /// \throws     nothing
    ///
    /// \see        \a hpx::diagnostic_information(), \a hpx::get_error_host_name(),
    ///             \a hpx::get_error_process_id(), \a hpx::get_error_function_name(),
    ///             \a hpx::get_error_file_name(), \a hpx::get_error_line_number(),
    ///             \a hpx::get_error_os_thread(), \a hpx::get_error_thread_id(),
    ///             \a hpx::get_error_thread_description(),
    ///             \a hpx::get_error_backtrace(), \a hpx::get_error_env(),
    ///             \a hpx::get_error_what(), \a hpx::get_error_config(),
    ///             \a hpx::get_error_state()
    ///
    HPX_EXPORT error get_error(hpx::exception const& e);

    /// \copydoc get_error(hpx::exception const& e)
    HPX_EXPORT error get_error(hpx::error_code const& e);

    /// \cond NOINTERNAL
    HPX_EXPORT error get_error(std::exception_ptr const& e);
    /// \endcond

    /// \brief Return the hostname of the locality where the exception was
    ///        thrown.
    ///
    /// The function \a hpx::get_error_host_name can be used to extract the
    /// diagnostic information element representing the host name as stored in
    /// the given exception instance.
    ///
    /// \param e    The parameter \p e will be inspected for the requested
    ///             diagnostic information elements which have been stored at
    ///             the point where the exception was thrown. This parameter
    ///             can be one of the following types: \a hpx::exception_info,
    ///             \a hpx::error_code, \a std::exception, or
    ///             \a std::exception_ptr.
    ///
    /// \returns    The hostname of the locality where the exception was
    ///             thrown. If the exception instance does not hold
    ///             this information, the function will return and empty string.
    ///
    /// \throws     std#bad_alloc (if one of the required allocations fails)
    ///
    /// \see        \a hpx::diagnostic_information()
    ///             \a hpx::get_error_process_id(), \a hpx::get_error_function_name(),
    ///             \a hpx::get_error_file_name(), \a hpx::get_error_line_number(),
    ///             \a hpx::get_error_os_thread(), \a hpx::get_error_thread_id(),
    ///             \a hpx::get_error_thread_description(), \a hpx::get_error()
    ///             \a hpx::get_error_backtrace(), \a hpx::get_error_env(),
    ///             \a hpx::get_error_what(), \a hpx::get_error_config(),
    ///             \a hpx::get_error_state()
    ///
    HPX_EXPORT std::string get_error_host_name(hpx::exception_info const& xi);

    /// \cond NOINTERNAL
    template <typename E>
    std::string get_error_host_name(E const& e)
    {
        if (exception_info const* xi = get_exception_info(e))
            return get_error_host_name(*xi);
        return std::string();
    }
    /// \endcond

    /// \brief Return the (operating system) process id of the locality where
    ///        the exception was thrown.
    ///
    /// The function \a hpx::get_error_process_id can be used to extract the
    /// diagnostic information element representing the process id as stored in
    /// the given exception instance.
    ///
    /// \returns    The process id of the OS-process which threw the exception
    ///             If the exception instance does not hold
    ///             this information, the function will return 0.
    ///
    /// \param e    The parameter \p e will be inspected for the requested
    ///             diagnostic information elements which have been stored at
    ///             the point where the exception was thrown. This parameter
    ///             can be one of the following types: \a hpx::exception_info,
    ///             \a hpx::error_code, \a std::exception, or
    ///             \a std::exception_ptr.
    ///
    /// \throws     nothing
    ///
    /// \see        \a hpx::diagnostic_information(), \a hpx::get_error_host_name(),
    ///             \a hpx::get_error_function_name(),
    ///             \a hpx::get_error_file_name(), \a hpx::get_error_line_number(),
    ///             \a hpx::get_error_os_thread(), \a hpx::get_error_thread_id(),
    ///             \a hpx::get_error_thread_description(), \a hpx::get_error(),
    ///             \a hpx::get_error_backtrace(), \a hpx::get_error_env(),
    ///             \a hpx::get_error_what(), \a hpx::get_error_config(),
    ///             \a hpx::get_error_state()
    ///
    HPX_EXPORT std::int64_t get_error_process_id(hpx::exception_info const& xi);

    /// \cond NOINTERNAL
    template <typename E>
    std::int64_t get_error_process_id(E const& e)
    {
        if (exception_info const* xi = get_exception_info(e))
            return get_error_process_id(*xi);
        return -1;
    }
    /// \endcond

    /// \brief Return the environment of the OS-process at the point the
    ///        exception was thrown.
    ///
    /// The function \a hpx::get_error_env can be used to extract the
    /// diagnostic information element representing the environment of the
    /// OS-process collected at the point the exception was thrown.
    ///
    /// \returns    The environment from the point the exception was
    ///             thrown. If the exception instance does not hold
    ///             this information, the function will return an empty string.
    ///
    /// \param e    The parameter \p e will be inspected for the requested
    ///             diagnostic information elements which have been stored at
    ///             the point where the exception was thrown. This parameter
    ///             can be one of the following types: \a hpx::exception_info,
    ///             \a hpx::error_code, \a std::exception, or
    ///             \a std::exception_ptr.
    ///
    /// \throws     std#bad_alloc (if one of the required allocations fails)
    ///
    /// \see        \a hpx::diagnostic_information(), \a hpx::get_error_host_name(),
    ///             \a hpx::get_error_process_id(), \a hpx::get_error_function_name(),
    ///             \a hpx::get_error_file_name(), \a hpx::get_error_line_number(),
    ///             \a hpx::get_error_os_thread(), \a hpx::get_error_thread_id(),
    ///             \a hpx::get_error_thread_description(), \a hpx::get_error(),
    ///             \a hpx::get_error_backtrace(),
    ///             \a hpx::get_error_what(), \a hpx::get_error_config(),
    ///             \a hpx::get_error_state()
    ///
    HPX_EXPORT std::string get_error_env(hpx::exception_info const& xi);

    /// \cond NOINTERNAL
    template <typename E>
    std::string get_error_env(E const& e)
    {
        if (exception_info const* xi = get_exception_info(e))
            return get_error_env(*xi);
        return "<unknown>";
    }
    /// \endcond

    /// \brief Return the function name from which the exception was thrown.
    ///
    /// The function \a hpx::get_error_function_name can be used to extract the
    /// diagnostic information element representing the name of the function
    /// as stored in the given exception instance.
    ///
    /// \returns    The name of the function from which the exception was
    ///             thrown. If the exception instance does not hold
    ///             this information, the function will return an empty string.
    ///
    /// \param e    The parameter \p e will be inspected for the requested
    ///             diagnostic information elements which have been stored at
    ///             the point where the exception was thrown. This parameter
    ///             can be one of the following types: \a hpx::exception_info,
    ///             \a hpx::error_code, \a std::exception, or
    ///             \a std::exception_ptr.
    ///
    /// \throws     std#bad_alloc (if one of the required allocations fails)
    ///
    /// \see        \a hpx::diagnostic_information(), \a hpx::get_error_host_name(),
    ///             \a hpx::get_error_process_id()
    ///             \a hpx::get_error_file_name(), \a hpx::get_error_line_number(),
    ///             \a hpx::get_error_os_thread(), \a hpx::get_error_thread_id(),
    ///             \a hpx::get_error_thread_description(), \a hpx::get_error(),
    ///             \a hpx::get_error_backtrace(), \a hpx::get_error_env(),
    ///             \a hpx::get_error_what(), \a hpx::get_error_config(),
    ///             \a hpx::get_error_state()
    ///
    HPX_EXPORT std::string get_error_function_name(hpx::exception_info const& xi);

    /// \cond NOINTERNAL
    template <typename E>
    std::string get_error_function_name(E const& e)
    {
        if (exception_info const* xi = get_exception_info(e))
            return get_error_function_name(*xi);
        return std::string();
    }
    /// \endcond

    /// \brief Return the stack backtrace from the point the exception was thrown.
    ///
    /// The function \a hpx::get_error_backtrace can be used to extract the
    /// diagnostic information element representing the stack backtrace
    /// collected at the point the exception was thrown.
    ///
    /// \returns    The stack back trace from the point the exception was
    ///             thrown. If the exception instance does not hold
    ///             this information, the function will return an empty string.
    ///
    /// \param e    The parameter \p e will be inspected for the requested
    ///             diagnostic information elements which have been stored at
    ///             the point where the exception was thrown. This parameter
    ///             can be one of the following types: \a hpx::exception_info,
    ///             \a hpx::error_code, \a std::exception, or
    ///             \a std::exception_ptr.
    ///
    /// \throws     std#bad_alloc (if one of the required allocations fails)
    ///
    /// \see        \a hpx::diagnostic_information(), \a hpx::get_error_host_name(),
    ///             \a hpx::get_error_process_id(), \a hpx::get_error_function_name(),
    ///             \a hpx::get_error_file_name(), \a hpx::get_error_line_number(),
    ///             \a hpx::get_error_os_thread(), \a hpx::get_error_thread_id(),
    ///             \a hpx::get_error_thread_description(), \a hpx::get_error(),
    ///             \a hpx::get_error_env(),
    ///             \a hpx::get_error_what(), \a hpx::get_error_config(),
    ///             \a hpx::get_error_state()
    ///
    HPX_EXPORT std::string get_error_backtrace(hpx::exception_info const& xi);

    /// \cond NOINTERNAL
    template <typename E>
    std::string get_error_backtrace(E const& e)
    {
        if (exception_info const* xi = get_exception_info(e))
            return get_error_backtrace(*xi);
        return std::string();
    }
    /// \endcond

    /// \brief Return the (source code) file name of the function from which
    ///        the exception was thrown.
    ///
    /// The function \a hpx::get_error_file_name can be used to extract the
    /// diagnostic information element representing the name of the source file
    /// as stored in the given exception instance.
    ///
    /// \returns    The name of the source file of the function from which the
    ///             exception was thrown. If the exception instance does
    ///             not hold this information, the function will return an empty
    ///             string.
    ///
    /// \param e    The parameter \p e will be inspected for the requested
    ///             diagnostic information elements which have been stored at
    ///             the point where the exception was thrown. This parameter
    ///             can be one of the following types: \a hpx::exception_info,
    ///             \a hpx::error_code, \a std::exception, or
    ///             \a std::exception_ptr.
    ///
    /// \throws     std#bad_alloc (if one of the required allocations fails)
    ///
    /// \see        \a hpx::diagnostic_information(), \a hpx::get_error_host_name(),
    ///             \a hpx::get_error_process_id(), \a hpx::get_error_function_name(),
    ///             \a hpx::get_error_line_number(),
    ///             \a hpx::get_error_os_thread(), \a hpx::get_error_thread_id(),
    ///             \a hpx::get_error_thread_description(), \a hpx::get_error(),
    ///             \a hpx::get_error_backtrace(), \a hpx::get_error_env(),
    ///             \a hpx::get_error_what(), \a hpx::get_error_config(),
    ///             \a hpx::get_error_state()
    ///
    HPX_EXPORT std::string get_error_file_name(hpx::exception_info const& xi);

    /// \cond NOINTERNAL
    template <typename E>
    std::string get_error_file_name(E const& e)
    {
        if (exception_info const* xi = get_exception_info(e))
            return get_error_file_name(*xi);
        return "<unknown>";
    }
    /// \endcond

    /// \brief Return the line number in the (source code) file of the function
    ///        from which the exception was thrown.
    ///
    /// The function \a hpx::get_error_line_number can be used to extract the
    /// diagnostic information element representing the line number
    /// as stored in the given exception instance.
    ///
    /// \returns    The line number of the place where the exception was
    ///             thrown. If the exception instance does not hold
    ///             this information, the function will return -1.
    ///
    /// \param e    The parameter \p e will be inspected for the requested
    ///             diagnostic information elements which have been stored at
    ///             the point where the exception was thrown. This parameter
    ///             can be one of the following types: \a hpx::exception_info,
    ///             \a hpx::error_code, \a std::exception, or
    ///             \a std::exception_ptr.
    ///
    /// \throws     nothing
    ///
    /// \see        \a hpx::diagnostic_information(), \a hpx::get_error_host_name(),
    ///             \a hpx::get_error_process_id(), \a hpx::get_error_function_name(),
    ///             \a hpx::get_error_file_name()
    ///             \a hpx::get_error_os_thread(), \a hpx::get_error_thread_id(),
    ///             \a hpx::get_error_thread_description(), \a hpx::get_error(),
    ///             \a hpx::get_error_backtrace(), \a hpx::get_error_env(),
    ///             \a hpx::get_error_what(), \a hpx::get_error_config(),
    ///             \a hpx::get_error_state()
    ///
    HPX_EXPORT long get_error_line_number(hpx::exception_info const& xi);

    /// \cond NOINTERNAL
    template <typename E>
    long get_error_line_number(E const& e)
    {
        if (exception_info const* xi = get_exception_info(e))
            return get_error_line_number(*xi);
        return -1;
    }
    /// \endcond

    /// \brief Return the sequence number of the OS-thread used to execute
    ///        HPX-threads from which the exception was thrown.
    ///
    /// The function \a hpx::get_error_os_thread can be used to extract the
    /// diagnostic information element representing the sequence number  of the
    /// OS-thread as stored in the given exception instance.
    ///
    /// \returns    The sequence number of the OS-thread used to execute the
    ///             HPX-thread from which the exception was
    ///             thrown. If the exception instance does not hold
    ///             this information, the function will return std::size(-1).
    ///
    /// \param e    The parameter \p e will be inspected for the requested
    ///             diagnostic information elements which have been stored at
    ///             the point where the exception was thrown. This parameter
    ///             can be one of the following types: \a hpx::exception_info,
    ///             \a hpx::error_code, \a std::exception, or
    ///             \a std::exception_ptr.
    ///
    /// \throws     nothing
    ///
    /// \see        \a hpx::diagnostic_information(), \a hpx::get_error_host_name(),
    ///             \a hpx::get_error_process_id(), \a hpx::get_error_function_name(),
    ///             \a hpx::get_error_file_name(), \a hpx::get_error_line_number(),
    ///             \a hpx::get_error_thread_id(),
    ///             \a hpx::get_error_thread_description(), \a hpx::get_error(),
    ///             \a hpx::get_error_backtrace(), \a hpx::get_error_env(),
    ///             \a hpx::get_error_what(), \a hpx::get_error_config(),
    ///             \a hpx::get_error_state()
    ///
    HPX_EXPORT std::size_t get_error_os_thread(hpx::exception_info const& xi);

    /// \cond NOINTERNAL
    template <typename E>
    std::size_t get_error_os_thread(E const& e)
    {
        if (exception_info const* xi = get_exception_info(e))
            return get_error_os_thread(*xi);
        return std::size_t(-1);
    }
    /// \endcond

    /// \brief Return the unique thread id of the HPX-thread from which the
    ///        exception was thrown.
    ///
    /// The function \a hpx::get_error_thread_id can be used to extract the
    /// diagnostic information element representing the HPX-thread id
    /// as stored in the given exception instance.
    ///
    /// \returns    The unique thread id of the HPX-thread from which the
    ///             exception was thrown. If the exception instance
    ///             does not hold this information, the function will return
    ///             std::size_t(0).
    ///
    /// \param e    The parameter \p e will be inspected for the requested
    ///             diagnostic information elements which have been stored at
    ///             the point where the exception was thrown. This parameter
    ///             can be one of the following types: \a hpx::exception_info,
    ///             \a hpx::error_code, \a std::exception, or
    ///             \a std::exception_ptr.
    ///
    /// \throws     nothing
    ///
    /// \see        \a hpx::diagnostic_information(), \a hpx::get_error_host_name(),
    ///             \a hpx::get_error_process_id(), \a hpx::get_error_function_name(),
    ///             \a hpx::get_error_file_name(), \a hpx::get_error_line_number(),
    ///             \a hpx::get_error_os_thread()
    ///             \a hpx::get_error_thread_description(), \a hpx::get_error(),
    ///             \a hpx::get_error_backtrace(), \a hpx::get_error_env(),
    ///             \a hpx::get_error_what(), \a hpx::get_error_config(),
    ///             \a hpx::get_error_state()
    ///
    HPX_EXPORT std::size_t get_error_thread_id(hpx::exception_info const& xi);

    /// \cond NOINTERNAL
    template <typename E>
    std::size_t get_error_thread_id(E const& e)
    {
        if (exception_info const* xi = get_exception_info(e))
            return get_error_thread_id(*xi);
        return std::size_t(-1);
    }
    /// \endcond

    /// \brief Return any additionally available thread description of the
    ///        HPX-thread from which the exception was thrown.
    ///
    /// The function \a hpx::get_error_thread_description can be used to extract the
    /// diagnostic information element representing the additional thread
    /// description as stored in the given exception instance.
    ///
    /// \returns    Any additionally available thread description of the
    ///             HPX-thread from which the exception was
    ///             thrown. If the exception instance does not hold
    ///             this information, the function will return an empty string.
    ///
    /// \param e    The parameter \p e will be inspected for the requested
    ///             diagnostic information elements which have been stored at
    ///             the point where the exception was thrown. This parameter
    ///             can be one of the following types: \a hpx::exception_info,
    ///             \a hpx::error_code, \a std::exception, or
    ///             \a std::exception_ptr.
    ///
    /// \throws     std#bad_alloc (if one of the required allocations fails)
    ///
    /// \see        \a hpx::diagnostic_information(), \a hpx::get_error_host_name(),
    ///             \a hpx::get_error_process_id(), \a hpx::get_error_function_name(),
    ///             \a hpx::get_error_file_name(), \a hpx::get_error_line_number(),
    ///             \a hpx::get_error_os_thread(), \a hpx::get_error_thread_id(),
    ///             \a hpx::get_error_backtrace(), \a hpx::get_error_env(),
    ///             \a hpx::get_error(), \a hpx::get_error_state(),
    ///             \a hpx::get_error_what(), \a hpx::get_error_config()
    ///
    HPX_EXPORT std::string get_error_thread_description(hpx::exception_info const& xi);

    /// \cond NOINTERNAL
    template <typename E>
    std::string get_error_thread_description(E const& e)
    {
        if (exception_info const* xi = get_exception_info(e))
            return get_error_thread_description(*xi);
        return std::string();
    }
    /// \endcond

    /// \brief Return the HPX configuration information point from which the
    ///        exception was thrown.
    ///
    /// The function \a hpx::get_error_config can be used to extract the
    /// HPX configuration information element representing the full HPX
    /// configuration information as stored in the given exception instance.
    ///
    /// \returns    Any additionally available HPX configuration information
    ///             the point from which the exception was
    ///             thrown. If the exception instance does not hold
    ///             this information, the function will return an empty string.
    ///
    /// \param e    The parameter \p e will be inspected for the requested
    ///             diagnostic information elements which have been stored at
    ///             the point where the exception was thrown. This parameter
    ///             can be one of the following types: \a hpx::exception_info,
    ///             \a hpx::error_code, \a std::exception, or
    ///             \a std::exception_ptr.
    ///
    /// \throws     std#bad_alloc (if one of the required allocations fails)
    ///
    /// \see        \a hpx::diagnostic_information(), \a hpx::get_error_host_name(),
    ///             \a hpx::get_error_process_id(), \a hpx::get_error_function_name(),
    ///             \a hpx::get_error_file_name(), \a hpx::get_error_line_number(),
    ///             \a hpx::get_error_os_thread(), \a hpx::get_error_thread_id(),
    ///             \a hpx::get_error_backtrace(), \a hpx::get_error_env(),
    ///             \a hpx::get_error(), \a hpx::get_error_state()
    ///             \a hpx::get_error_what(), \a hpx::get_error_thread_description()
    ///
    HPX_EXPORT std::string get_error_config(hpx::exception_info const& xi);

    /// \cond NOINTERNAL
    template <typename E>
    std::string get_error_config(E const& e)
    {
        if (exception_info const* xi = get_exception_info(e))
            return get_error_config(*xi);
        return std::string();
    }
    /// \endcond

    /// \brief Return the HPX runtime state information at which the exception
    ///        was thrown.
    ///
    /// The function \a hpx::get_error_state can be used to extract the
    /// HPX runtime state information element representing the state the
    /// runtime system is currently in as stored in the given exception
    /// instance.
    ///
    /// \returns    The point runtime state at the point at which the exception
    ///             was thrown. If the exception instance does not hold
    ///             this information, the function will return an empty string.
    ///
    /// \param e    The parameter \p e will be inspected for the requested
    ///             diagnostic information elements which have been stored at
    ///             the point where the exception was thrown. This parameter
    ///             can be one of the following types: \a hpx::exception_info,
    ///             \a hpx::error_code, \a std::exception, or
    ///             \a std::exception_ptr.
    ///
    /// \throws     std#bad_alloc (if one of the required allocations fails)
    ///
    /// \see        \a hpx::diagnostic_information(), \a hpx::get_error_host_name(),
    ///             \a hpx::get_error_process_id(), \a hpx::get_error_function_name(),
    ///             \a hpx::get_error_file_name(), \a hpx::get_error_line_number(),
    ///             \a hpx::get_error_os_thread(), \a hpx::get_error_thread_id(),
    ///             \a hpx::get_error_backtrace(), \a hpx::get_error_env(),
    ///             \a hpx::get_error(),
    ///             \a hpx::get_error_what(), \a hpx::get_error_thread_description()
    ///
    HPX_EXPORT std::string get_error_state(hpx::exception_info const& xi);

    /// \cond NOINTERNAL
    template <typename E>
    std::string get_error_state(E const& e)
    {
        if (exception_info const* xi = get_exception_info(e))
            return get_error_state(*xi);
        return std::string();
    }
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    // \cond NOINTERNAL
    // forwarder for HPX_ASSERT handler
    HPX_ATTRIBUTE_NORETURN HPX_EXPORT
    void assertion_failed(char const* expr, char const* function,
        char const* file, long line);

    // forwarder for HPX_ASSERT_MSG handler
    HPX_ATTRIBUTE_NORETURN HPX_EXPORT
    void assertion_failed_msg(char const* msg, char const* expr,
        char const* function, char const* file, long line);

    // For testing purposes we sometime expect to see exceptions, allow those
    // to go through without attaching a debugger.
    //
    // This should be used carefully as it disables the possible attaching of
    // a debugger for all exceptions, not only the expected ones.
    HPX_EXPORT bool expect_exception(bool flag = true);
    /// \endcond
}

#include <hpx/throw_exception.hpp>

#include <hpx/config/warnings_suffix.hpp>

#endif
