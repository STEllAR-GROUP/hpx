//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file exception_fwd.hpp

#ifndef HPX_EXCEPTION_FWD_HPP
#define HPX_EXCEPTION_FWD_HPP

#include <hpx/config.hpp>
#include <hpx/error.hpp>

#include <boost/exception_ptr.hpp>

/// \cond NOINTERNAL
namespace boost
{
    class exception_ptr;
}
/// \endcond

namespace hpx
{
    /// \cond NOINTERNAL
    // forward declaration
    class error_code;
    class exception;
    /// \endcond

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

    ///////////////////////////////////////////////////////////////////////////
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
    ///             can be one of the following types: \a hpx::exception,
    ///             \a hpx::error_code, \a boost::exception, or
    ///             \a boost::exception_ptr.
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
    HPX_EXPORT std::string get_error_what(hpx::exception const& e);

    /// \copydoc get_error_locality_id(hpx::exception const& e)
    HPX_EXPORT std::string get_error_what(hpx::error_code const& e);

    /// \cond NOINTERNAL
    HPX_EXPORT std::string get_error_what(boost::exception const& e);
    HPX_EXPORT std::string get_error_what(boost::exception_ptr const& e);
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
    ///             \a hpx::error_code, or \a boost::exception_ptr.
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
    HPX_EXPORT error get_error(boost::exception_ptr const& e);
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    // \cond NOINTERNAL
    HPX_EXPORT boost::exception_ptr get_exception_ptr(hpx::exception const& e);
    /// \endcond
}

#include <hpx/throw_exception.hpp>

#endif
