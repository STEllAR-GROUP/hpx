//  Copyright (c) 2019 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  Make HPX inspect tool happy:
//                               hpxinspect:noinclude:HPX_ASSERT
//                               hpxinspect:noinclude:HPX_ASSERT_MSG

//  Note: There are no include guards. This is intentional.

#include <hpx/config.hpp>
#include <hpx/assertion/current_function.hpp>
#include <hpx/assertion/evaluate_assert.hpp>
#include <hpx/assertion/source_location.hpp>
#include <hpx/preprocessor/stringize.hpp>

#include <string>
#include <type_traits>

namespace hpx { namespace assertion {
    /// The signature for an assertion handler
    using assertion_handler = void (*)(
        source_location const& loc, const char* expr, std::string const& msg);

    /// Set the assertion handler to be used within a program. If the handler has been
    /// set already once, the call to this function will be ignored.
    /// \note This function is not thread safe
    HPX_EXPORT void set_assertion_handler(assertion_handler handler);
}}    // namespace hpx::assertion

#if defined(DOXYGEN)
/// \def HPX_ASSERT(expr, msg)
/// \brief This macro asserts that \a expr evaluates to true.
///
/// \param expr The expression to assert on. This can either be an expression
///             that's convertible to bool or a callable which returns bool
/// \param msg The optional message that is used to give further information if
///             the assert fails. This should be convertible to a std::string
///
/// If \a expr evaluates to false, The source location and \a msg is being
/// printed along with the expression and additional. Afterwards the program is
/// being aborted. The assertion handler can be costumized by calling \a
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
    ::hpx::assertion::detail::evaluate_assert(                                 \
        [&]() -> bool { return !!(expr); },                                    \
        ::hpx::assertion::source_location{__FILE__,                            \
            static_cast<unsigned>(__LINE__), HPX_ASSERT_CURRENT_FUNCTION},     \
        HPX_PP_STRINGIZE(expr), [&]() { return msg; });                        \
    /**/

#if defined(HPX_DEBUG)
#define HPX_ASSERT(expr) HPX_ASSERT_(expr, std::string())
#define HPX_ASSERT_MSG(expr, msg) HPX_ASSERT_(expr, msg)
#else
#define HPX_ASSERT(expr)                                                       \
    {                                                                          \
    }
#define HPX_ASSERT_MSG(expr, msg)                                              \
    {                                                                          \
    }
#endif
#endif
