//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if \
 !defined(HPX_RUNTIME_PARCELSET_POLICIES_COALESCING_MESSAGE_HANDLER_FEB_24_2013_0302PM)
#define HPX_RUNTIME_PARCELSET_POLICIES_COALESCING_MESSAGE_HANDLER_FEB_24_2013_0302PM

#include <hpx/hpx_fwd.hpp>

#if defined(HPX_HAVE_PARCEL_COALESCING)

#include <hpx/runtime/parcelset/policies/message_handler.hpp>
#include <hpx/util/interval_timer.hpp>
#include <hpx/util/detail/count_num_args.hpp>
#include <hpx/lcos/local/spinlock.hpp>

#include <hpx/plugins/parcel/message_buffer.hpp>

#include <boost/preprocessor/stringize.hpp>
#include <boost/thread/locks.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace plugins { namespace parcel
{
    struct HPX_LIBRARY_EXPORT coalescing_message_handler
      : parcelset::policies::message_handler
    {
    private:
        coalescing_message_handler* this_() { return this; }

        typedef lcos::local::spinlock mutex_type;

    public:
        typedef parcelset::policies::message_handler::write_handler_type
            write_handler_type;

        coalescing_message_handler(char const* action_name,
            parcelset::parcelport* pp, std::size_t num = std::size_t(-1),
            std::size_t interval = std::size_t(-1));

        void put_parcel(parcelset::locality const & dest,
            parcelset::parcel p, write_handler_type f);

        bool flush(bool stop_buffering = false);

    protected:
        bool timer_flush();
        bool flush(boost::unique_lock<mutex_type>& l, bool stop_buffering);

    private:
        mutable mutex_type mtx_;
        parcelset::parcelport* pp_;
        detail::message_buffer buffer_;
        util::interval_timer timer_;
        bool stopped_;
    };
}}}

#include <hpx/config/warnings_suffix.hpp>

///////////////////////////////////////////////////////////////////////////////
#define HPX_ACTION_USES_MESSAGE_COALESCING(...)                               \
    HPX_ACTION_USES_MESSAGE_COALESCING_(__VA_ARGS__)                          \
/**/

#define HPX_ACTION_USES_MESSAGE_COALESCING_(...)                              \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_ACTION_USES_MESSAGE_COALESCING_, HPX_UTIL_PP_NARG(__VA_ARGS__)    \
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_ACTION_USES_MESSAGE_COALESCING_1(action_type)                     \
    HPX_ACTION_USES_MESSAGE_COALESCING_4(action_type,                         \
        BOOST_PP_STRINGIZE(action_type), std::size_t(-1), std::size_t(-1))    \
/**/

#define HPX_ACTION_USES_MESSAGE_COALESCING_2(action_type, num)                \
    HPX_ACTION_USES_MESSAGE_COALESCING_3(action_type,                         \
        BOOST_PP_STRINGIZE(action_type), num, std::size_t(-1))                \
/**/

#define HPX_ACTION_USES_MESSAGE_COALESCING_3(action_type, num, interval)      \
    HPX_ACTION_USES_MESSAGE_COALESCING_3(action_type,                         \
        BOOST_PP_STRINGIZE(action_type), num, interval)                       \
/**/

#define HPX_ACTION_USES_MESSAGE_COALESCING_4(                                 \
        action_type, action_name, num, interval)                              \
    namespace hpx { namespace traits                                          \
    {                                                                         \
        template <>                                                           \
        struct action_message_handler<action_type>                            \
        {                                                                     \
            static parcelset::policies::message_handler* call(                \
                parcelset::parcelhandler* ph, parcelset::locality const& loc, \
                parcelset::parcel const& /*p*/)                               \
            {                                                                 \
                return parcelset::get_message_handler(ph, action_name,        \
                    "coalescing_message_handler", num, interval, loc);        \
            }                                                                 \
        };                                                                    \
    }}                                                                        \
/**/

#define HPX_ACTION_USES_MESSAGE_COALESCING_NOTHROW(                           \
        action_type, action_name, num, interval)                              \
    namespace hpx { namespace traits                                          \
    {                                                                         \
        template <>                                                           \
        struct action_message_handler<action_type>                            \
        {                                                                     \
            static parcelset::policies::message_handler* call(                \
                parcelset::parcelhandler* ph, parcelset::locality const& loc, \
                parcelset::parcel const& /*p*/)                               \
            {                                                                 \
                error_code ec(lightweight);                                   \
                return parcelset::get_message_handler(ph, action_name,        \
                    "coalescing_message_handler", num, interval, loc, ec);    \
            }                                                                 \
        };                                                                    \
    }}                                                                        \
/**/

#else

#define HPX_ACTION_USES_MESSAGE_COALESCING(...)
#define HPX_ACTION_USES_MESSAGE_COALESCING_NOTHROW(...)

#endif

#endif
