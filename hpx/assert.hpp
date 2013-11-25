//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file assert.hpp

#include <hpx/config/branch_hints.hpp>

//--------------------------------------------------------------------------------------//
//                                     HPX_ASSERT                                     //
//--------------------------------------------------------------------------------------//

#undef HPX_ASSERT

#if defined(HPX_DISABLE_ASSERTS)

# define HPX_ASSERT(expr) ((void)0)

#elif defined(HPX_ENABLE_ASSERT_HANDLER)

#include <boost/config.hpp>
#include <boost/current_function.hpp>

namespace hpx
{
  void assertion_failed(char const * expr,
                        char const * function, char const * file, long line); // user defined
} // namespace hpx

#define HPX_ASSERT(expr) (HPX_LIKELY(!!(expr)) \
  ? ((void)0) \
  : ::hpx::assertion_failed(#expr, BOOST_CURRENT_FUNCTION, __FILE__, __LINE__))

#else
# include <assert.h> // .h to support old libraries w/o <cassert> - effect is the same
# define HPX_ASSERT(expr) assert(expr)
#endif

//--------------------------------------------------------------------------------------//
//                                   HPX_ASSERT_MSG                                   //
//--------------------------------------------------------------------------------------//

# undef HPX_ASSERT_MSG

#if defined(HPX_DISABLE_ASSERTS) || defined(NDEBUG)

  #define HPX_ASSERT_MSG(expr, msg) ((void)0)

#elif defined(HPX_ENABLE_ASSERT_HANDLER)

  #include <boost/config.hpp>
  #include <boost/current_function.hpp>

  namespace hpx
  {
    void assertion_failed_msg(char const * expr, char const * msg,
                              char const * function, char const * file, long line); // user defined
  } // namespace hpx

  #define HPX_ASSERT_MSG(expr, msg) (HPX_LIKELY(!!(expr)) \
    ? ((void)0) \
    : ::hpx::assertion_failed_msg(#expr, msg, BOOST_CURRENT_FUNCTION, __FILE__, __LINE__))

#else
  #ifndef HPX_ASSERT_HPP
    #define HPX_ASSERT_HPP
    #include <cstdlib>
    #include <iostream>
    #include <boost/config.hpp>
    #include <boost/current_function.hpp>

    //  IDE's like Visual Studio perform better if output goes to std::cout or
    //  some other stream, so allow user to configure output stream:
    #ifndef HPX_ASSERT_MSG_OSTREAM
    # define HPX_ASSERT_MSG_OSTREAM std::cerr
    #endif

    namespace hpx
    {
      namespace assertion
      {
        namespace detail
        {
          // Note: The template is needed to make the function non-inline and avoid linking errors
          template< typename CharT >
          BOOST_NOINLINE void assertion_failed_msg(CharT const * expr, char const * msg, char const * function,
            char const * file, long line)
          {
            HPX_ASSERT_MSG_OSTREAM
              << "***** Internal Program Error - assertion (" << expr << ") failed in "
              << function << ":\n"
              << file << '(' << line << "): " << msg << std::endl;
#ifdef UNDER_CE
            // The Windows CE CRT library does not have abort() so use exit(-1) instead.
            std::exit(-1);
#else
            std::abort();
#endif
          }
        } // detail
      } // assertion
    } // detail
  #endif

  #define HPX_ASSERT_MSG(expr, msg) (HPX_LIKELY(!!(expr)) \
    ? ((void)0) \
    : ::hpx::assertion::detail::assertion_failed_msg(#expr, msg, \
          BOOST_CURRENT_FUNCTION, __FILE__, __LINE__))
#endif

//--------------------------------------------------------------------------------------//
//                                     HPX_VERIFY                                     //
//--------------------------------------------------------------------------------------//

#undef HPX_VERIFY

#if defined(HPX_DISABLE_ASSERTS) || ( !defined(HPX_ENABLE_ASSERT_HANDLER) && defined(NDEBUG) )

# define HPX_VERIFY(expr) ((void)(expr))

#else

# define HPX_VERIFY(expr) HPX_ASSERT(expr)

#endif
