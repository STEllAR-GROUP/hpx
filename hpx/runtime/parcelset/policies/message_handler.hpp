//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_PARCELSET_POLICIES_MESSAGE_HANDLER_FEB_24_2013_1141AM)
#define HPX_RUNTIME_PARCELSET_POLICIES_MESSAGE_HANDLER_FEB_24_2013_1141AM

#include <hpx/config.hpp>
#include <hpx/exception_fwd.hpp>
#include <hpx/runtime/parcelset_fwd.hpp>
#include <hpx/util/function.hpp>

namespace hpx { namespace parcelset { namespace policies
{
    struct message_handler
    {
        enum flush_mode
        {
            flush_mode_timer = 0,
            flush_mode_background_work = 1,
            flush_mode_buffer_full = 2
        };

        typedef util::function_nonser<
            void(boost::system::error_code const&, parcel const&)
        > write_handler_type;

        virtual ~message_handler() {}
        virtual void put_parcel(
            parcelset::locality const& dest, parcel p,
            write_handler_type f) = 0;
        virtual bool flush(flush_mode mode, bool stop_buffering = false) = 0;
    };
}}}

#endif
