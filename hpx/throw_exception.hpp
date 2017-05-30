//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file throw_exception.hpp

#ifndef HPX_THROW_EXCEPTION_HPP
#define HPX_THROW_EXCEPTION_HPP

#include <hpx/config.hpp>
#include <hpx/compat/exception.hpp>
#include <hpx/error.hpp>
#include <hpx/exception_fwd.hpp>

#include <boost/current_function.hpp>
#include <boost/system/error_code.hpp>

#include <string>

#include <hpx/config/warnings_prefix.hpp>

/// \cond NODETAIL
namespace hpx { namespace detail
{
    template <typename Exception>
    HPX_ATTRIBUTE_NORETURN HPX_EXPORT
    void throw_exception(Exception const& e,
        std::string const& func, std::string const& file, long line);

    HPX_ATTRIBUTE_NORETURN HPX_EXPORT void throw_exception(
        error errcode, std::string const& msg,
        std::string const& func, std::string const& file, long line);

    HPX_ATTRIBUTE_NORETURN HPX_EXPORT void rethrow_exception(
        exception const& e, std::string const& func);

    template <typename Exception>
    HPX_EXPORT compat::exception_ptr get_exception(Exception const& e,
            std::string const& func = "<unknown>",
            std::string const& file = "<unknown>",
            long line = -1,
            std::string const& auxinfo = "");

    HPX_EXPORT compat::exception_ptr get_exception(
            error errcode, std::string const& msg, throwmode mode,
            std::string const& func = "<unknown>",
            std::string const& file = "<unknown>",
            long line = -1,
            std::string const& auxinfo = "");

    HPX_EXPORT compat::exception_ptr get_exception(
            boost::system::error_code ec, std::string const& msg, throwmode mode,
            std::string const& func = "<unknown>",
            std::string const& file = "<unknown>",
            long line = -1,
            std::string const& auxinfo = "");

    HPX_EXPORT void throws_if(
        hpx::error_code& ec, error errcode, std::string const& msg,
        std::string const& func, std::string const& file, long line);

    HPX_EXPORT void rethrows_if(
        hpx::error_code& ec, exception const& e, std::string const& func);

    HPX_ATTRIBUTE_NORETURN HPX_EXPORT
    void throw_thread_interrupted_exception();
}}
/// \endcond

namespace hpx
{
    /// \cond NOINTERNAL

    /// \brief throw an hpx::exception initialized from the given arguments
    HPX_ATTRIBUTE_NORETURN inline
    void throw_exception(error e, std::string const& msg,
        std::string const& func, std::string const& file = "", long line = -1)
    {
        detail::throw_exception(e, msg, func, file, line);
    }
    /// \endcond
}

/// \cond NOINTERNAL
///////////////////////////////////////////////////////////////////////////////
// helper macro allowing to prepend file name and line number to a generated
// exception
#define HPX_THROW_STD_EXCEPTION(except, func)                                 \
    hpx::detail::throw_exception(except, func, __FILE__, __LINE__)            \
    /**/

#define HPX_RETHROW_EXCEPTION(e, f)                                           \
    hpx::detail::rethrow_exception(e, f)                                      \
    /**/

#define HPX_RETHROWS_IF(ec, e, f)                                             \
    hpx::detail::rethrows_if(ec, e, f)                                        \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_GET_EXCEPTION(errcode, f, msg)                                    \
    hpx::detail::get_exception(errcode, msg, hpx::plain, f,                   \
        __FILE__, __LINE__)                                                   \
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
    HPX_THROWS_IF(ec, errcode, BOOST_CURRENT_FUNCTION, msg)                   \
    /**/

#define HPX_RETHROWS_IN_CURRENT_FUNC_IF(ec, errcode, msg)                     \
    HPX_RETHROWS_IF(ec, errcode, BOOST_CURRENT_FUNCTION, msg)                 \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_THROW_THREAD_INTERRUPTED_EXCEPTION()                              \
    hpx::detail::throw_thread_interrupted_exception()                         \
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
    hpx::detail::throw_exception(errcode, msg, f, __FILE__, __LINE__)         \
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
    hpx::detail::throws_if(ec, errcode, msg, f, __FILE__, __LINE__)           \
    /**/

#include <hpx/config/warnings_suffix.hpp>

#endif
