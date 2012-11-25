//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_SHMEM_MESSAGE_NOV_25_2012_0847PM)
#define HPX_PARCELSET_SHMEM_MESSAGE_NOV_25_2012_0847PM

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace shmem
{
    struct message
    {
        enum commands
        {
            connect = 1,
            shutdown = 2,
        };

        commands command_;
        char data_[32];
    };
}}}

#endif

