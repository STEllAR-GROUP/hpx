//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_ACTION_MESSAGE_HANDLER_FEB_24_2013_0318PM)
#define HPX_TRAITS_ACTION_MESSAGE_HANDLER_FEB_24_2013_0318PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/parcelset/policies/message_handler.hpp>

#include <boost/mpl/has_xxx.hpp>
#include <boost/utility/enable_if.hpp>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    // Customization point for action stack size
    template <typename Action, typename Enable>
    struct action_message_handler
    {
        // return a new instance of a serialization filter
        static parcelset::policies::message_handler* call()
        {
            return 0;   // by default actions don't have a message_handler
        }
    };

    template <typename Action>
    struct action_message_handler<Action, typename Action::type>
      : action_message_handler<typename Action::type>
    {};
}}

#endif

