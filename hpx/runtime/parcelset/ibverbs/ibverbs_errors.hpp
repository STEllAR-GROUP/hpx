//  Copyright (c)      2013 Thomas Heller
//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_IBVERBS_ERRORS_HPP)
#define HPX_PARCELSET_IBVERBS_ERRORS_HPP

#include <hpx/config.hpp>
#include <boost/interprocess/errors.hpp>
#include <boost/system/system_error.hpp>
#include <boost/static_assert.hpp>

#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace ibverbs
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        char const* const error_names[] =
        {
            "foo"
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
            static_cast<int>(e), get_interprocess_category());
    }
    */

}}}

#define HPX_IBVERBS_THROWS_IF(ec, code)                                         \
    if (&ec != &boost::system::throws) ec = code;                               \
    else boost::asio::detail::throw_error(code);                                \
/**/

#define HPX_IBVERBS_RESET_EC(ec)                                                \
    if (&ec != &boost::system::throws) ec = boost::system::error_code();        \
/**/

#define HPX_IBVERBS_NEXT_WC(ec, state, ret, retry)                              \
    {                                                                           \
        message_type m = next_wc(ec, retry);                                    \
        if(ec) return ret;                                                      \
        if(m == MSG_RETRY && !retry)                                            \
        {                                                                       \
            return ret;                                                         \
        }                                                                       \
        if(m != state)                                                          \
        {                                                                       \
            HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::fault);               \
            return ret;                                                         \
        }                                                                       \
    }
/**/

#endif
