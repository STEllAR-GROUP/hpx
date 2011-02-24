////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_F646702C_6556_48FA_BF9D_3E7959983122)
#define HPX_F646702C_6556_48FA_BF9D_3E7959983122

#include <boost/detail/lightweight_test.hpp>

namespace hpx { namespace detail
{

inline void sanity_failed_impl(char const* expr, char const* file, int line,
                             char const* function)
{
    BOOST_LIGHTWEIGHT_TEST_OSTREAM
      << file << "(" << line << "): sanity check '"
      << expr << "' failed in function '"
      << function << "'" << std::endl;
        ++::boost::detail::test_errors();
}

template <typename T, typename U>
inline void sanity_eq_impl(char const* expr1, char const* expr2,
                           char const* file, int line, char const* function,
                           T const& t, U const& u)
{
    if (!(t == u))
    {
        BOOST_LIGHTWEIGHT_TEST_OSTREAM
            << file << "(" << line << "): sanity check '" << expr1
            << " == " << expr2
            << "' failed in function '" << function << "': "
            << "'" << t << "' != '" << u << "'" << std::endl;
        ++::boost::detail::test_errors();
    }
}

template <typename T, typename U>
inline void test_msg_impl(char const* msg, char const* file, int line, 
                          char const* function, T const& t, U const& u)
{
    if (!(t == u))
    {
        ::boost::detail::error_impl(msg, file, line, function);
    }
}

} // detail
} // hpx

#define HPX_TEST(expr)            BOOST_TEST(expr)
#define HPX_ERROR(msg)            BOOST_ERROR(msg)
#define HPX_TEST_EQ(expr1, expr2) BOOST_TEST_EQ(expr1, expr2)

#define HPX_TEST_MSG(expr, msg)                           \
  ((expr)                                                 \
   ? (void)0                                              \
   : ::boost::detail::error_impl                          \
      (msg, __FILE__, __LINE__, BOOST_CURRENT_FUNCTION))  \
  /***/

#define HPX_TEST_EQ_MSG(expr1, expr2, msg)    \
  (::hpx::detail::test_msg_impl               \
    (msg, __FILE__, __LINE__,                 \
     BOOST_CURRENT_FUNCTION, expr1, expr2))   \
  /***/

#define HPX_SANITY(expr)                                   \
  ((expr)                                                  \
   ? (void)0                                               \
   : ::hpx::detail::sanity_failed_impl                     \
      (#expr, __FILE__, __LINE__, BOOST_CURRENT_FUNCTION)) \
  /***/

#define HPX_SANITY_EQ(expr1,expr2)          \
  (::hpx::detail::sanity_eq_impl            \
    (#expr1, #expr2, __FILE__, __LINE__,    \
     BOOST_CURRENT_FUNCTION, expr1, expr2)) \
  /***/

#endif // HPX_F646702C_6556_48FA_BF9D_3E7959983122

