//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_TRAITS_ACTION_MESSAGE_HANDLER_HPP
#define HPX_TRAITS_ACTION_MESSAGE_HANDLER_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/parcelset_fwd.hpp>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    // Customization point for action stack size
    template <typename Action, typename Enable>
    struct action_message_handler
    {
        // return a new instance of a serialization filter
        static parcelset::policies::message_handler* call(
            parcelset::parcelhandler*, parcelset::locality const&,
            parcelset::parcel const&)
        {
            return 0;   // by default actions don't have a message_handler
        }
    };
}}

#endif /*HPX_TRAITS_ACTION_MESSAGE_HANDLER_HPP*/
