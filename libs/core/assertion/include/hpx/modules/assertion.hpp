//  Copyright (c) 2019 Thomas Heller
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  Make HPX inspect tool happy:
//                               hpxinspect:noinclude:HPX_ASSERT
//                               hpxinspect:noinclude:HPX_ASSERT_MSG
//                               hpxinspect:noassert_macro

#pragma once

#include <hpx/config.hpp>
#include <hpx/assertion/evaluate_assert.hpp>
#include <hpx/assertion/source_location.hpp>
#include <hpx/preprocessor/stringize.hpp>

#if defined(HPX_COMPUTE_DEVICE_CODE)
#include <assert.h>
#endif
#include <cstdint>
#include <exception>
#include <string>
#include <type_traits>

namespace hpx::assertion {

    /// The signature for an assertion handler
    using assertion_handler = void (*)(hpx::source_location const& loc,
        const char* expr, std::string const& msg);

    /// Set the assertion handler to be used within a program. If the handler has been
    /// set already once, the call to this function will be ignored.
    /// \note This function is not thread safe
    HPX_CORE_EXPORT void set_assertion_handler(assertion_handler handler);
}    // namespace hpx::assertion

#if defined(DOXYGEN)
/// \def HPX_ASSERT(expr, msg)
/// \brief This macro asserts that \a expr evaluates to true.
///
/// \param expr The expression to assert on. This can either be an expression
///             that's convertible to bool or a callable which returns bool
/// \param msg The optional message that is used to give further information if
///             the assert fails. This should be convertible to a std::string
///
/// If \p expr evaluates to false, The source location and \p msg is being
/// printed along with the expression and additional. Afterwards the program is
/// being aborted. The assertion handler can be customized by calling
/// hpx::assertion::set_assertion_handler().
///
/// Asserts are enabled if \a HPX_DEBUG is set. This is the default for
/// `CMAKE_BUILD_TYPE=Debug`
#define HPX_ASSERT(expr)

/// \see HPX_ASSERT
#define HPX_ASSERT_MSG(expr, msg)
#else
/// \cond NOINTERNAL
#define HPX_ASSERT_(expr, msg)                                                 \
    (!!(expr) ? void() :                                                       \
                ::hpx::assertion::detail::handle_assert(                       \
                    HPX_CURRENT_SOURCE_LOCATION(), HPX_PP_STRINGIZE(expr),     \
                    msg)) /**/

#define HPX_ASSERT_LOCKED_(l, expr, msg)                                       \
    (!!(expr) ? void() :                                                       \
                (l.unlock(),                                                   \
                    ::hpx::assertion::detail::handle_assert(                   \
                        HPX_CURRENT_SOURCE_LOCATION(), HPX_PP_STRINGIZE(expr), \
                        msg))) /**/

#if defined(HPX_DEBUG)
#if defined(HPX_COMPUTE_DEVICE_CODE)
#define HPX_ASSERT(expr) assert(expr)
#define HPX_ASSERT_MSG(expr, msg) HPX_ASSERT(expr)
#define HPX_ASSERT_LOCKED(l, expr) assert(expr)
#define HPX_ASSERT_LOCKED_MSG(l, expr, msg) HPX_ASSERT(expr)
#else
#define HPX_ASSERT(expr) HPX_ASSERT_(expr, std::string())
#define HPX_ASSERT_MSG(expr, msg) HPX_ASSERT_(expr, msg)
#define HPX_ASSERT_LOCKED(l, expr) HPX_ASSERT_LOCKED_(l, expr, std::string())
#define HPX_ASSERT_LOCKED_MSG(l, expr, msg) HPX_ASSERT_LOCKED_(l, expr, msg)
#endif
#else
#define HPX_ASSERT(expr)
#define HPX_ASSERT_MSG(expr, msg)
#define HPX_ASSERT_LOCKED(l, expr)
#define HPX_ASSERT_LOCKED_MSG(l, expr, msg)
#endif

#define HPX_UNREACHABLE                                                        \
    HPX_ASSERT_(false,                                                         \
        "This code is meant to be unreachable. If you are seeing this error "  \
        "message it means that you have found a bug in HPX. Please report it " \
        "on the issue tracker: https://github.com/STEllAR-GROUP/hpx/issues."); \
    std::terminate()
#endif
