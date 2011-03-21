////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_F646702C_6556_48FA_BF9D_3E7959983122)
#define HPX_F646702C_6556_48FA_BF9D_3E7959983122

#include <cstddef>

#include <iostream>

#include <boost/config.hpp>
#include <boost/current_function.hpp>
#include <boost/preprocessor/stringize.hpp>

namespace hpx { namespace util { namespace detail
{

std::size_t sanity_failures = 0;
std::size_t test_failures = 0;

template <typename T>
inline bool check(char const* file, int line, char const* function,
                  std::size_t& counter, T const& t, char const* msg)
{
    if (!t)
    { 
        std::cerr 
            << file << "(" << line << "): "
            << msg << " failed in function '"
            << function << "'" << std::endl;
        ++counter;
        return false;
    }
    return true;
}

template <typename T, typename U>
inline bool check_eq(char const* file, int line, char const* function,
                     std::size_t& counter, T const& t, U const& u,
                     char const* msg)
{
    if (!(t == u))
    {
        std::cerr 
            << file << "(" << line << "): " << msg  
            << " failed in function '" << function << "': "
            << "'" << t << "' != '" << u << "'" << std::endl;
        ++counter;
        return false;
    }
    return true;
}

} // hpx::util::detail

inline int report_errors()
{
    if ((detail::sanity_failures == 0) && (detail::test_failures == 0))
        return 0;

    else
    {
        std::cerr << detail::sanity_failures << " sanity check"
                  << ((detail::sanity_failures == 1) ? " and " : "s and ")
                  << detail::test_failures << " test"
                  << ((detail::test_failures == 1) ? " failed." : "s failed.")
                  << std::endl;
        return 1;
    }
}

}} // hpx::util

#define HPX_TEST(expr)                                                      \
    ::hpx::util::detail::check                                              \
        (__FILE__, __LINE__, BOOST_CURRENT_FUNCTION,                        \
         ::hpx::util::detail::test_failures,                                \
         expr, "test '" BOOST_PP_STRINGIZE(expr) "'")                       \
    /***/

#define HPX_TEST_MSG(expr, msg)                                             \
    ::hpx::util::detail::check                                              \
        (__FILE__, __LINE__, BOOST_CURRENT_FUNCTION,                        \
         ::hpx::util::detail::test_failures,                                \
         expr, msg)                                                         \
    /***/

#define HPX_TEST_EQ(expr1, expr2)                                           \
    ::hpx::util::detail::check_eq                                           \
        (__FILE__, __LINE__, BOOST_CURRENT_FUNCTION,                        \
         ::hpx::util::detail::test_failures,                                \
         expr1, expr2, "test '" BOOST_PP_STRINGIZE(expr1) " == "            \
                                BOOST_PP_STRINGIZE(expr2) "'")              \
    /***/

#define HPX_TEST_EQ_MSG(expr1, expr2, msg)                                  \
    ::hpx::util::detail::check_eq                                           \
        (__FILE__, __LINE__, BOOST_CURRENT_FUNCTION,                        \
         ::hpx::util::detail::test_failures,                                \
         expr1, expr2, msg)                                                 \
    /***/

#define HPX_SANITY(expr)                                                    \
    ::hpx::util::detail::check                                              \
        (__FILE__, __LINE__, BOOST_CURRENT_FUNCTION,                        \
         ::hpx::util::detail::sanity_failures,                              \
         expr, "sanity check '" BOOST_PP_STRINGIZE(expr) "'")               \
    /***/

#define HPX_SANITY_MSG(expr, msg)                                           \
    ::hpx::util::detail::check                                              \
        (__FILE__, __LINE__, BOOST_CURRENT_FUNCTION,                        \
         ::hpx::util::detail::sanity_failures,                              \
         expr, msg)                                                         \
    /***/

#define HPX_SANITY_EQ(expr1, expr2)                                         \
    ::hpx::util::detail::check_eq                                           \
        (__FILE__, __LINE__, BOOST_CURRENT_FUNCTION,                        \
         ::hpx::util::detail::sanity_failures,                              \
         expr1, expr2, "sanity check '" BOOST_PP_STRINGIZE(expr1) " == "    \
                                        BOOST_PP_STRINGIZE(expr2) "'")      \
    /***/

#define HPX_SANITY_EQ_MSG(expr1, expr2, msg)                                \
    ::hpx::util::detail::check_eq                                           \
        (__FILE__, __LINE__, BOOST_CURRENT_FUNCTION,                        \
         ::hpx::util::detail::sanity_failures,                              \
         expr1, expr2)                                                      \
    /***/

#endif // HPX_F646702C_6556_48FA_BF9D_3E7959983122

