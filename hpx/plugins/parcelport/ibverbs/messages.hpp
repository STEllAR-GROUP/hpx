//  Copyright (c) 2013-2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_POLICIES_IBVERBS_MESSAGES_HPP)
#define HPX_PARCELSET_POLICIES_IBVERBS_MESSAGES_HPP

#include <hpx/config/defines.hpp>
#if defined(HPX_HAVE_PARCELPORT_IBVERBS)

#include <hpx/config.hpp>

namespace hpx { namespace parcelset { namespace policies { namespace ibverbs {

    enum message_type
    {
        MSG_INVALID = 0,
        MSG_RETRY = 1,
        MSG_MR = 2,
        MSG_SIZE = 3,
        MSG_DATA = 4,
        MSG_DONE = 5,
        MSG_SHUTDOWN = 6
    };

    struct message
    {
    private:
        HPX_NON_COPYABLE(message);

    public:
        message() {}

        boost::uint32_t id;
        boost::uint32_t rkey;
        boost::uint64_t addr;
        boost::uint64_t size;

        static const std::size_t data_size =
            sizeof(boost::uint32_t)*2 + sizeof(boost::uint64_t)*2;
        static const std::size_t payload_size =
            HPX_WITH_PARCELPORT_IBVERBS_MESSAGE_PAYLOAD - data_size;

        char payload[payload_size];
    };
}}}}

#endif

#endif
