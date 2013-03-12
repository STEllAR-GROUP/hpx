//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_PARCELSET_POLICIES_COALESCING_MESSAGE_HANDLER_FEB_24_2013_0302PM)
#define HPX_RUNTIME_PARCELSET_POLICIES_COALESCING_MESSAGE_HANDLER_FEB_24_2013_0302PM

#include <hpx/hpx_fwd.hpp>

#if defined(HPX_HAVE_PARCEL_COALESCING)

#include <hpx/runtime/parcelset/policies/message_handler.hpp>
#include <hpx/util/interval_timer.hpp>
#include <hpx/plugins/parcel/message_buffer.hpp>

#include <boost/preprocessor/stringize.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace plugins { namespace parcel
{
    struct HPX_LIBRARY_EXPORT coalescing_message_handler
      : parcelset::policies::message_handler
    {
    private:
        coalescing_message_handler* this_() { return this; }

    public:
        typedef parcelset::policies::message_handler::write_handler_type
            write_handler_type;

        coalescing_message_handler(char const* action_name, 
            parcelset::parcelport* set, std::size_t num, std::size_t interval = 100);
        ~coalescing_message_handler();

        void put_parcel(parcelset::parcel& p, write_handler_type const& f);

        void flush(bool stop_buffering = false);

    protected:
        bool timer_flush();

    private:
        parcelset::parcelport* pp_;
        detail::message_buffer buffer_;
        util::interval_timer timer_;
        boost::int64_t interval_;
        bool stopped_;
    };
}}}

#include <hpx/config/warnings_suffix.hpp>

///////////////////////////////////////////////////////////////////////////////
#define HPX_ACTION_USES_MESSAGE_COALESCING(action_type, num)                  \
    namespace hpx { namespace traits                                          \
    {                                                                         \
        template <>                                                           \
        struct action_message_handler<action_type>                            \
        {                                                                     \
            static parcelset::policies::message_handler* call(                \
                parcelset::parcelhandler* ph, naming::locality const& loc,    \
                parcelset::connection_type t)                                 \
            {                                                                 \
                return ph->get_message_handler<                               \
                    hpx::plugins::parcel::coalescing_message_handler          \
                >(BOOST_PP_STRINGIZE(action_type), num, loc, t);              \
            }                                                                 \
        };                                                                    \
    }}                                                                        \
/**/

#else

#define HPX_ACTION_USES_MESSAGE_COALESCING(action_type, num)

#endif

#endif
