//  Copyright (c) 2013 Antoine Tran Tan
//  Copyright (c) 2001, 2002 Peter Dimov and Multi Media Ltd.
//  Copyright (c) 2007 Peter Dimov
//  Copyright (c) Beman Dawes 2011
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  Make HPX inspect tool happy: hpxinspect:noassert_macro
//  Note: There are no include guards. This is intentional.

#if defined(HPX_USE_BOOST_ASSERT)
#   include <boost/assert.hpp>
#   define HPX_ASSERT(expr) BOOST_ASSERT(expr)
#   define HPX_ASSERT_MSG(expr, msg) BOOST_ASSERT_MSG(expr, msg)
#else

#include <hpx/config.hpp>

//-------------------------------------------------------------------------- //
//                                     HPX_ASSERT                            //
//-------------------------------------------------------------------------- //

#undef HPX_ASSERT

#if defined(__CUDA_ARCH__)
#define HPX_ASSERT(expr) ((void)0)
#else

#if defined(HPX_DISABLE_ASSERTS) || defined(BOOST_DISABLE_ASSERTS) || defined(NDEBUG)

#if defined(HPX_GCC_VERSION) || defined(HPX_CLANG_VERSION)
# define HPX_ASSERT(expr) ((expr) ? (void)0 : __builtin_unreachable())
#elif defined(HPX_MSVC) && !defined(HPX_INTEL_WIN)
# define HPX_ASSERT(expr) __assume(!!(expr))
#else
# define HPX_ASSERT(expr) ((void)0)
#endif

#elif defined(HPX_ENABLE_ASSERT_HANDLER) || defined(BOOST_ENABLE_ASSERT_HANDLER)

#include <hpx/config.hpp>
#include <boost/current_function.hpp>

namespace hpx
{
    HPX_NORETURN HPX_EXPORT void assertion_failed(char const * expr,
        char const * function, char const * file, long line);  // user defined
} // namespace hpx

#define HPX_ASSERT(expr) (HPX_LIKELY(!!(expr)) \
  ? ((void)0) \
  : ::hpx::assertion_failed(#expr, BOOST_CURRENT_FUNCTION, __FILE__, __LINE__))

#else
# include <assert.h> // .h to support old libraries w/o <cassert> - effect is the same
# define HPX_ASSERT(expr) assert(expr)
#endif
#endif

//-------------------------------------------------------------------------- //
//                                   HPX_ASSERT_MSG                          //
//-------------------------------------------------------------------------- //

# undef HPX_ASSERT_MSG

#if defined(__CUDA_ARCH__)
#define HPX_ASSERT_MSG(expr) ((void)0)
#else

#if defined(HPX_DISABLE_ASSERTS) || defined(BOOST_DISABLE_ASSERTS) || defined(NDEBUG)

#if defined(HPX_GCC_VERSION) || defined(HPX_CLANG_VERSION)
# define HPX_ASSERT_MSG(expr, msg) ((expr) ? (void)0 : __builtin_unreachable())
#elif defined(HPX_MSVC) && !defined(HPX_INTEL_WIN)
# define HPX_ASSERT_MSG(expr, msg) __assume(!!(expr))
#else
# define HPX_ASSERT_MSG(expr, msg) ((void)0)
#endif

#elif defined(HPX_ENABLE_ASSERT_HANDLER) || defined(BOOST_ENABLE_ASSERT_HANDLER)

#include <hpx/config.hpp>
#include <boost/current_function.hpp>

namespace hpx
{
    HPX_NORETURN HPX_EXPORT void assertion_failed_msg(
        char const * expr, char const * msg,
        char const * function, char const * file, long line); // user defined
} // namespace hpx

#define HPX_ASSERT_MSG(expr, msg) (HPX_LIKELY(!!(expr)) \
    ? ((void)0) \
    : ::hpx::assertion_failed_msg \
    (#expr, msg, BOOST_CURRENT_FUNCTION, __FILE__, __LINE__))

#else

#ifndef HPX_ASSERT_HPP
#define HPX_ASSERT_HPP

#include <hpx/config.hpp>
#include <boost/current_function.hpp>
#include <cstdlib>
#include <iostream>

//  IDE's like Visual Studio perform better if output goes to std::cout or
//  some other stream, so allow user to configure output stream:
#ifndef HPX_ASSERT_MSG_OSTREAM
# define HPX_ASSERT_MSG_OSTREAM std::cerr
#endif

namespace hpx { namespace assertion { namespace detail
{
    // Note: The template is needed to make the function non-inline and
    // avoid linking errors
    template <typename CharT>
    HPX_NORETURN HPX_NOINLINE void assertion_failed_msg(
        CharT const * expr, char const * msg, char const * function,
        char const * file, long line)
    {
        HPX_ASSERT_MSG_OSTREAM
            << "***** Internal Program Error - assertion (" << expr
            << ") failed in " << function << ":\n"
            << file << '(' << line << "): " << msg << std::endl;
#ifdef UNDER_CE
        // The Windows CE CRT library does not have abort() so use exit(-1) instead.
        std::exit(-1);
#else
        std::abort();
#endif
    }
}}}

#endif

#define HPX_ASSERT_MSG(expr, msg) (HPX_LIKELY(!!(expr)) \
    ? ((void)0) \
    : ::hpx::assertion::detail::assertion_failed_msg(#expr, msg, \
          BOOST_CURRENT_FUNCTION, __FILE__, __LINE__))
#endif
#endif

//---------------------------------------------------------------------------//
//                                     HPX_VERIFY                            //
//---------------------------------------------------------------------------//

#undef HPX_VERIFY

#if defined(HPX_DISABLE_ASSERTS) || ( !defined(HPX_ENABLE_ASSERT_HANDLER) \
 && defined(NDEBUG) ) || defined(BOOST_DISABLE_ASSERTS) \
 || ( !defined(BOOST_ENABLE_ASSERT_HANDLER) && defined(NDEBUG) )

# define HPX_VERIFY(expr) ((void)(expr))

#else

# define HPX_VERIFY(expr) HPX_ASSERT(expr)

#endif

#endif
