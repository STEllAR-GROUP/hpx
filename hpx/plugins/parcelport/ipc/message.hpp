//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_IPC_MESSAGE_NOV_25_2012_0847PM)
#define HPX_PARCELSET_IPC_MESSAGE_NOV_25_2012_0847PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_PARCELPORT_IPC)

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace policies { namespace ipc
{
    struct message
    {
        enum commands
        {
            connect = 1,
            data = 2,
            acknowledge = 3,
            shutdown = 4,
        };

        commands command_;
        char data_[33];
    };
}}}}

#endif

#endif

