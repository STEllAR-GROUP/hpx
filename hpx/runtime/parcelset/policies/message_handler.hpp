//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_PARCELSET_POLICIES_MESSAGE_HANDLER_FEB_24_2013_1141AM)
#define HPX_RUNTIME_PARCELSET_POLICIES_MESSAGE_HANDLER_FEB_24_2013_1141AM

#include <hpx/hpx_fwd.hpp>

namespace hpx { namespace parcelset { namespace policies
{
    struct message_handler
    {
        typedef HPX_STD_FUNCTION<
            void(boost::system::error_code const&, std::size_t)
        > write_handler_type;

        virtual ~message_handler() {}
        virtual void put_parcel(parcel& p, write_handler_type const& f) = 0;
        virtual void flush(bool stop_buffering = false) = 0;
    };
}}}

#endif
