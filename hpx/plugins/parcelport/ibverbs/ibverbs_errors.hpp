//  Copyright (c)      2013 Thomas Heller
//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_POLICIES_IBVERBS_ERRORS_HPP)
#define HPX_PARCELSET_POLICIES_IBVERBS_ERRORS_HPP

#include <hpx/config/defines.hpp>
#if defined(HPX_HAVE_PARCELPORT_IBVERBS)

#include <hpx/config.hpp>
#include <boost/system/system_error.hpp>
#include <boost/asio/error.hpp>
#include <boost/asio/detail/throw_error.hpp>
#include <boost/static_assert.hpp>

#include <string>

#if !defined(BOOST_SYSTEM_NOEXCEPT)
#define BOOST_SYSTEM_NOEXCEPT BOOST_NOEXCEPT
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace policies { namespace ibverbs
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        char const* const error_names[] =
        {
            "unkown_error"
        };

        class ibverbs_category : public boost::system::error_category
        {
        public:
            const char* name() const BOOST_SYSTEM_NOEXCEPT
            {
                return "ibverbs";
            }

            std::string message(int value) const
            {
                return "ibverbs(unkown_error)";
            }
        };
    }

    inline boost::system::error_category const & get_ibverbs_category()
    {
        static detail::ibverbs_category instance;
        return instance;
    }

    ///////////////////////////////////////////////////////////////////////////
    /*
    inline boost::system::error_code
    make_error_code(boost::interprocess::error_code_t e)
    {
        return boost::system::error_code(
            static_cast<int>(e), get_ibverbs_category());
    }
    */
}}}}

#define HPX_IBVERBS_THROWS_IF(ec, code)                                         \
    if (&ec != &boost::system::throws) ec = code;                               \
    else boost::asio::detail::throw_error(code);                                \
/**/

#define HPX_IBVERBS_THROWS(code)                                                \
    boost::asio::detail::throw_error(code)                                      \
/**/

#define HPX_IBVERBS_RESET_EC(ec)                                                \
    if (&ec != &boost::system::throws) ec = boost::system::error_code();        \
/**/

#define HPX_IBVERBS_NEXT_WC(ec, state, N, ret, retry)                           \
    BOOST_PP_CAT(HPX_IBVERBS_NEXT_WC_, retry) (ec, state, N, ret)               \
/**/

#define HPX_IBVERBS_NEXT_WC_true(ec, state, N, ret)                             \
    {                                                                           \
        message_type m = next_wc<true, N>::call(this, ec);                      \
        if(ec) return ret;                                                      \
        if(m == MSG_SHUTDOWN)                                                   \
        {                                                                       \
            close_locked(ec);                                                   \
            HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::eof);                 \
            return ret;                                                         \
        }                                                                       \
        if(m != state)                                                          \
        {                                                                       \
            HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::fault);               \
            return ret;                                                         \
        }                                                                       \
    }
/**/

#define HPX_IBVERBS_NEXT_WC_false(ec, state, N, ret)                            \
    {                                                                           \
        message_type m = next_wc<false, N>::call(this, ec);                     \
        if(ec) return ret;                                                      \
        if(m == MSG_SHUTDOWN)                                                   \
        {                                                                       \
            close_locked(ec);                                                   \
            HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::eof);                 \
            return ret;                                                         \
        }                                                                       \
        if(m == MSG_RETRY)                                                      \
        {                                                                       \
            return ret;                                                         \
        }                                                                       \
        if(m != state)                                                          \
        {                                                                       \
            HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::fault);               \
            return ret;                                                         \
        }                                                                       \
    }

#endif

#endif
