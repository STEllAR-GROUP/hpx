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
        typedef util::function_nonser<
            void(boost::system::error_code const&, parcel const&)
        > write_handler_type;

        virtual ~message_handler() {}
        virtual void put_parcel(parcelset::locality const & dest, parcel p,
            write_handler_type f) = 0;
        virtual bool flush(bool stop_buffering = false) = 0;
    };
}}}

#endif
