//  Copyright (c)      2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_IBVERBS_MESSAGES_HPP)
#define HPX_PARCELSET_IBVERBS_MESSAGES_HPP

namespace hpx { namespace parcelset { namespace ibverbs {
        
    enum message_type
    {
        MSG_INVALID = 0,
        MSG_RETRY = 1,
        MSG_MR = 2,
        MSG_READY = 3,
        MSG_DATA = 4,
        MSG_DONE = 5
    };

    struct message
    {
        int id;
        boost::uint64_t addr;
        boost::uint32_t rkey;
    };
}}}

#endif
