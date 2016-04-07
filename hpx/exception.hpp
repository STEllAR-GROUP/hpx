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

#include <boost/cstdint.hpp>
#include <boost/exception_ptr.hpp>
#include <boost/throw_exception.hpp>
#include <boost/system/error_code.hpp>
#include <boost/system/system_error.hpp>

#include <cstddef>
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
        ~exception();

        /// The function \a get_error() returns the hpx::error code stored
        /// in the referenced instance of a hpx::exception. It returns
        /// the hpx::error code this exception instance was constructed
        /// from.
        ///
        /// \throws nothing
        error get_error() const HPX_NOEXCEPT;

        /// The function \a get_error_code() returns a hpx::error_code which
        /// represents the same error condition as this hpx::exception instance.
        ///
        /// \param mode   The parameter \p mode specifies whether the returned
        ///               hpx::error_code belongs to the error category
        ///               \a hpx_category (if mode is \a plain, this is the
        ///               default) or to the category \a hpx_category_rethrow
        ///               (if mode is \a rethrow).
        error_code get_error_code(throwmode mode = plain) const HPX_NOEXCEPT;
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
        struct tag_throw_config {};
        struct tag_throw_state {};
        struct tag_throw_auxinfo {};

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

        // Stores the full HPX configuration information of the locality the
        // exception has been raised in. This information will show up in error
        // messages under the [config] tag.
        typedef boost::error_info<detail::tag_throw_config, std::string>
            throw_config;

        // Stores the current runtime state. This information will show up in
        // error messages under the [state] tag.
        typedef boost::error_info<detail::tag_throw_state, std::string>
            throw_state;

        // Stores additional auxiliary information (such as information about
        // the current parcel). This information will show up in error messages
        // under the [auxinfo] tag.
        typedef boost::error_info<detail::tag_throw_auxinfo, std::string>
            throw_auxinfo;

        // construct an exception, internal helper
        template <typename Exception>
        HPX_EXPORT boost::exception_ptr
            construct_exception(Exception const& e,
                std::string const& func, std::string const& file, long line,
                std::string const& back_trace = "", boost::uint32_t node = 0,
                std::string const& hostname = "", boost::int64_t pid = -1,
                std::size_t shepherd = ~0, std::size_t thread_id = 0,
                std::string const& thread_name = "",
                std::string const& env = "", std::string const& config = "",
                std::string const& state = "", std::string const& auxinfo = "");

        template <typename Exception>
        HPX_EXPORT boost::exception_ptr
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
        HPX_EXPORT void report_exception_and_terminate(boost::exception_ptr const&);
        HPX_EXPORT void report_exception_and_terminate(hpx::exception const&);

        // Report an early or late exception and locally exit execution. There
        // isn't anything more we could do. The exception will be re-thrown
        // from hpx::init
        HPX_EXPORT void report_exception_and_continue(boost::exception_ptr const&);
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
    ///             of the following types: \a hpx::exception or
    ///             \a hpx::error_code.
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
    HPX_EXPORT std::string diagnostic_information(hpx::exception const& e);

    /// \copydoc diagnostic_information(hpx::exception const& e)
    HPX_EXPORT std::string diagnostic_information(hpx::error_code const& e);

    /// \cond NOINTERNAL
    HPX_EXPORT std::string diagnostic_information(boost::exception const& e);
    HPX_EXPORT std::string diagnostic_information(boost::exception_ptr const& e);
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
    ///             can be one of the following types: \a hpx::exception,
    ///             \a hpx::error_code, \a boost::exception, or
    ///             \a boost::exception_ptr.
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
    HPX_EXPORT boost::uint32_t get_error_locality_id(hpx::exception const& e);

    /// \copydoc get_error_locality_id(hpx::exception const& e)
    HPX_EXPORT boost::uint32_t get_error_locality_id(hpx::error_code const& e);

    /// \cond NOINTERNAL
    HPX_EXPORT boost::uint32_t get_error_locality_id(boost::exception const& e);
    HPX_EXPORT boost::uint32_t get_error_locality_id(boost::exception_ptr const& e);
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
    ///             can be one of the following types: \a hpx::exception,
    ///             \a hpx::error_code, \a boost::exception, or
    ///             \a boost::exception_ptr.
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
    HPX_EXPORT std::string get_error_host_name(hpx::exception const& e);

    /// \copydoc get_error_host_name(hpx::exception const& e)
    HPX_EXPORT std::string get_error_host_name(hpx::error_code const& e);

    /// \cond NOINTERNAL
    HPX_EXPORT std::string get_error_host_name(boost::exception const& e);
    HPX_EXPORT std::string get_error_host_name(boost::exception_ptr const& e);
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
    ///             can be one of the following types: \a hpx::exception,
    ///             \a hpx::error_code, \a boost::exception, or
    ///             \a boost::exception_ptr.
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
    HPX_EXPORT boost::int64_t get_error_process_id(hpx::exception const& e);

    /// \copydoc get_error_process_id(hpx::exception const& e)
    HPX_EXPORT boost::int64_t get_error_process_id(hpx::error_code const& e);

    /// \cond NOINTERNAL
    HPX_EXPORT boost::int64_t get_error_process_id(boost::exception const& e);
    HPX_EXPORT boost::int64_t get_error_process_id(boost::exception_ptr const& e);
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
    ///             can be one of the following types: \a hpx::exception,
    ///             \a hpx::error_code, \a boost::exception, or
    ///             \a boost::exception_ptr.
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
    HPX_EXPORT std::string get_error_env(hpx::exception const& e);

    /// \copydoc get_error_env(hpx::exception const& e)
    HPX_EXPORT std::string get_error_env(hpx::error_code const& e);

    /// \cond NOINTERNAL
    HPX_EXPORT std::string get_error_env(boost::exception const& e);
    HPX_EXPORT std::string get_error_env(boost::exception_ptr const& e);
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
    ///             can be one of the following types: \a hpx::exception,
    ///             \a hpx::error_code, \a boost::exception, or
    ///             \a boost::exception_ptr.
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
    HPX_EXPORT std::string get_error_function_name(hpx::exception const& e);

    /// \copydoc get_error_function_name(hpx::exception const& e)
    HPX_EXPORT std::string get_error_function_name(hpx::error_code const& e);

    /// \cond NOINTERNAL
    HPX_EXPORT std::string get_error_function_name(boost::exception const& e);
    HPX_EXPORT std::string get_error_function_name(boost::exception_ptr const& e);
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
    ///             can be one of the following types: \a hpx::exception,
    ///             \a hpx::error_code, \a boost::exception, or
    ///             \a boost::exception_ptr.
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
    HPX_EXPORT std::string get_error_backtrace(hpx::exception const& e);

    /// \copydoc get_error_backtrace(hpx::exception const& e)
    HPX_EXPORT std::string get_error_backtrace(hpx::error_code const& e);

    /// \cond NOINTERNAL
    HPX_EXPORT std::string get_error_backtrace(boost::exception const& e);
    HPX_EXPORT std::string get_error_backtrace(boost::exception_ptr const& e);
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
    ///             can be one of the following types: \a hpx::exception,
    ///             \a hpx::error_code, \a boost::exception, or
    ///             \a boost::exception_ptr.
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
    HPX_EXPORT std::string get_error_file_name(hpx::exception const& e);

    /// \copydoc get_error_file_name(hpx::exception const& e)
    HPX_EXPORT std::string get_error_file_name(hpx::error_code const& e);

    /// \cond NOINTERNAL
    HPX_EXPORT std::string get_error_file_name(boost::exception const& e);
    HPX_EXPORT std::string get_error_file_name(boost::exception_ptr const& e);
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
    ///             can be one of the following types: \a hpx::exception,
    ///             \a hpx::error_code, \a boost::exception, or
    ///             \a boost::exception_ptr.
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
    HPX_EXPORT int get_error_line_number(hpx::exception const& e);

    /// \copydoc get_error_line_number(hpx::exception const& e)
    HPX_EXPORT int get_error_line_number(hpx::error_code const& e);

    /// \cond NOINTERNAL
    HPX_EXPORT int get_error_line_number(boost::exception const& e);
    HPX_EXPORT int get_error_line_number(boost::exception_ptr const& e);
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
    ///             can be one of the following types: \a hpx::exception,
    ///             \a hpx::error_code, \a boost::exception, or
    ///             \a boost::exception_ptr.
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
    HPX_EXPORT std::size_t get_error_os_thread(hpx::exception const& e);

    /// \copydoc get_error_os_thread(hpx::exception const& e)
    HPX_EXPORT std::size_t get_error_os_thread(hpx::error_code const& e);

    /// \cond NOINTERNAL
    HPX_EXPORT std::size_t get_error_os_thread(boost::exception const& e);
    HPX_EXPORT std::size_t get_error_os_thread(boost::exception_ptr const& e);
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
    ///             can be one of the following types: \a hpx::exception,
    ///             \a hpx::error_code, \a boost::exception, or
    ///             \a boost::exception_ptr.
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
    HPX_EXPORT std::size_t get_error_thread_id(hpx::exception const& e);

    /// \copydoc get_error_thread_id(hpx::exception const& e)
    HPX_EXPORT std::size_t get_error_thread_id(hpx::error_code const& e);

    /// \cond NOINTERNAL
    HPX_EXPORT std::size_t get_error_thread_id(boost::exception const& e);
    HPX_EXPORT std::size_t get_error_thread_id(boost::exception_ptr const& e);
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
    ///             can be one of the following types: \a hpx::exception,
    ///             \a hpx::error_code, \a boost::exception, or
    ///             \a boost::exception_ptr.
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
    HPX_EXPORT std::string get_error_thread_description(hpx::exception const& e);

    /// \copydoc get_error_thread_description(hpx::exception const& e)
    HPX_EXPORT std::string get_error_thread_description(hpx::error_code const& e);

    /// \cond NOINTERNAL
    HPX_EXPORT std::string get_error_thread_description(boost::exception const& e);
    HPX_EXPORT std::string get_error_thread_description(boost::exception_ptr const& e);
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
    ///             can be one of the following types: \a hpx::exception,
    ///             \a hpx::error_code, \a boost::exception, or
    ///             \a boost::exception_ptr.
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
    HPX_EXPORT std::string get_error_config(hpx::exception const& e);

    /// \copydoc get_error_config(hpx::exception const& e)
    HPX_EXPORT std::string get_error_config(hpx::error_code const& e);

    /// \cond NOINTERNAL
    HPX_EXPORT std::string get_error_config(boost::exception const& e);
    HPX_EXPORT std::string get_error_config(boost::exception_ptr const& e);
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
    ///             can be one of the following types: \a hpx::exception,
    ///             \a hpx::error_code, \a boost::exception, or
    ///             \a boost::exception_ptr.
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
    HPX_EXPORT std::string get_error_state(hpx::exception const& e);

    /// \copydoc get_error_state(hpx::exception const& e)
    HPX_EXPORT std::string get_error_state(hpx::error_code const& e);

    /// \cond NOINTERNAL
    HPX_EXPORT std::string get_error_state(boost::exception const& e);
    HPX_EXPORT std::string get_error_state(boost::exception_ptr const& e);
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
