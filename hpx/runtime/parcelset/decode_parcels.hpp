//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_DECODE_PARCELS_HPP
#define HPX_PARCELSET_DECODE_PARCELS_HPP

#include <hpx/config.hpp>
#include <hpx/util/portable_binary_archive.hpp>
#include <hpx/runtime/parcelset/decode_parcel.hpp>

#include <boost/shared_ptr.hpp>

#include <vector>

namespace hpx { namespace parcelset
{
    template <typename Parcelport, typename Connection, typename Buffer>
    void decode_parcels(Parcelport & parcelport, Connection & connection,
        Buffer buffer)
    {
        bool first_message = false;
#if defined(HPX_HAVE_SECURITY)
        if(connection.first_message_)
        {
            connection.first_message_ = false;
            first_message = true;
        }
#endif
        if(hpx::is_running() && parcelport.async_serialization())
        {
            hpx::applier::register_thread_nullary(
                util::bind(
                    util::one_shot(&decode_message<Parcelport, Buffer>),
                    boost::ref(parcelport), std::move(buffer), 0, first_message),
                "decode_parcels",
                threads::pending, true, threads::thread_priority_boost);
        }
        else
        {
            decode_message(parcelport, std::move(buffer), 0, first_message);
        }
    }

}}

#endif
