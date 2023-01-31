//  Copyright (c) 2007-2023 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file throw_exception.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/assertion/current_function.hpp>
#include <hpx/errors/error.hpp>
#include <hpx/errors/exception_fwd.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/preprocessor/cat.hpp>
#include <hpx/preprocessor/expand.hpp>
#include <hpx/preprocessor/nargs.hpp>

#include <exception>
#include <string>
#include <system_error>

#include <hpx/config/warnings_prefix.hpp>

/// \cond NODETAIL
namespace hpx::detail {

    template <typename Exception>
    [[noreturn]] HPX_CORE_EXPORT void throw_exception(Exception const& e,
        std::string const& func, std::string const& file, long line);

    [[noreturn]] HPX_CORE_EXPORT void throw_exception(error errcode,
        std::string const& msg, std::string const& func,
        std::string const& file, long line);

    [[noreturn]] HPX_CORE_EXPORT void rethrow_exception(
        exception const& e, std::string const& func);

    template <typename Exception>
    [[nodiscard]] HPX_CORE_EXPORT std::exception_ptr get_exception(
        Exception const& e, std::string const& func = "<unknown>",
        std::string const& file = "<unknown>", long line = -1,
        std::string const& auxinfo = "");

    [[nodiscard]] HPX_CORE_EXPORT std::exception_ptr get_exception(
        error errcode, std::string const& msg, throwmode mode,
        std::string const& func = "<unknown>",
        std::string const& file = "<unknown>", long line = -1,
        std::string const& auxinfo = "");

    [[nodiscard]] HPX_CORE_EXPORT std::exception_ptr get_exception(
        std::error_code const& ec, std::string const& msg, throwmode mode,
        std::string const& func = "<unknown>",
        std::string const& file = "<unknown>", long line = -1,
        std::string const& auxinfo = "");

    HPX_CORE_EXPORT void throws_if(hpx::error_code& ec, error errcode,
        std::string const& msg, std::string const& func,
        std::string const& file, long line);

    HPX_CORE_EXPORT void rethrows_if(
        hpx::error_code& ec, exception const& e, std::string const& func);

    [[noreturn]] HPX_CORE_EXPORT void throw_thread_interrupted_exception();
}    // namespace hpx::detail
/// \endcond

namespace hpx {
    /// \cond NOINTERNAL

    /// \brief throw an hpx::exception initialized from the given arguments
    [[noreturn]] inline void throw_exception(error e, std::string const& msg,
        std::string const& func, std::string const& file = "", long line = -1)
    {
        detail::throw_exception(e, msg, func, file, line);
    }
    /// \endcond
}    // namespace hpx

/// \cond NOINTERNAL
///////////////////////////////////////////////////////////////////////////////
// helper macro allowing to prepend file name and line number to a generated
// exception
#define HPX_THROW_STD_EXCEPTION(except, func)                                  \
    hpx::detail::throw_exception(except, func, __FILE__, __LINE__) /**/

#define HPX_RETHROW_EXCEPTION(e, f) hpx::detail::rethrow_exception(e, f) /**/

#define HPX_RETHROWS_IF(ec, e, f) hpx::detail::rethrows_if(ec, e, f) /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_GET_EXCEPTION(...)                                                 \
    HPX_GET_EXCEPTION_(__VA_ARGS__)                                            \
    /**/

#define HPX_GET_EXCEPTION_(...)                                                \
    HPX_PP_EXPAND(HPX_PP_CAT(HPX_GET_EXCEPTION_, HPX_PP_NARGS(__VA_ARGS__))(   \
        __VA_ARGS__))                                                          \
/**/
#define HPX_GET_EXCEPTION_3(errcode, f, msg)                                   \
    HPX_GET_EXCEPTION_4(errcode, hpx::throwmode::plain, f, msg)                \
/**/
#define HPX_GET_EXCEPTION_4(errcode, mode, f, msg)                             \
    hpx::detail::get_exception(errcode, msg, mode, f, __FILE__, __LINE__) /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_THROW_IN_CURRENT_FUNC(errcode, msg)                                \
    HPX_THROW_EXCEPTION(errcode, HPX_ASSERTION_CURRENT_FUNCTION, msg)          \
    /**/

#define HPX_RETHROW_IN_CURRENT_FUNC(errcode, msg)                              \
    HPX_RETHROW_EXCEPTION(errcode, HPX_ASSERTION_CURRENT_FUNCTION, msg)        \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_THROWS_IN_CURRENT_FUNC_IF(ec, errcode, msg)                        \
    HPX_THROWS_IF(ec, errcode, HPX_ASSERTION_CURRENT_FUNCTION, msg)            \
    /**/

#define HPX_RETHROWS_IN_CURRENT_FUNC_IF(ec, errcode, msg)                      \
    HPX_RETHROWS_IF(ec, errcode, HPX_ASSERTION_CURRENT_FUNCTION, msg)          \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_THROW_THREAD_INTERRUPTED_EXCEPTION()                               \
    hpx::detail::throw_thread_interrupted_exception() /**/
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
///          HPX_THROW_EXCEPTION(hpx::error::no_success, "raise_exception",
///             "simulated error");
///      }
/// \endcode
///
#define HPX_THROW_EXCEPTION(errcode, f, ...)                                   \
    hpx::detail::throw_exception(                                              \
        errcode, hpx::util::format(__VA_ARGS__), f, __FILE__, __LINE__) /**/

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
#define HPX_THROWS_IF(ec, errcode, f, ...)                                     \
    hpx::detail::throws_if(ec, errcode, hpx::util::format(__VA_ARGS__), f,     \
        __FILE__, __LINE__) /**/

#include <hpx/config/warnings_suffix.hpp>
