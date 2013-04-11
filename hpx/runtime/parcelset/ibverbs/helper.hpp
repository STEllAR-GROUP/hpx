//  Copyright (c)      2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_IBVERBS_HELPER_HPP)
#define HPX_PARCELSET_IBVERBS_HELPER_HPP


#include <netdb.h>
#include <rdma/rdma_cma.h>

#include <sys/time.h>
#include <sys/types.h>
#include <poll.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

namespace hpx { namespace parcelset { namespace ibverbs {
    template <typename Connection>
    inline bool get_next_event(
        rdma_event_channel *event_channel, rdma_cm_event & event_copy, Connection * c
      , boost::system::error_code &ec
    )
    {
        if(!event_channel)
        {
            HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::not_connected);
            return false;
        }

        rdma_cm_event * event = NULL;

        pollfd pfd;
        pfd.fd = event_channel->fd;
        pfd.events = POLLIN;
        pfd.revents = 0;

        int ret = 0;
        ret = poll(&pfd, 1, 1);
        if(ret == 0) return false;
        if(ret < 0)
        {
            int verrno = errno;
            boost::system::error_code err(verrno, boost::system::system_category());
            HPX_IBVERBS_THROWS_IF(
                ec
              , err
            );
            return false;
        }

        if(rdma_get_cm_event(event_channel, &event) == 0)
        {
            std::memcpy(&event_copy, event, sizeof(rdma_cm_event));

            rdma_ack_cm_event(event);

            if(event_copy.event == RDMA_CM_EVENT_DISCONNECTED)
            {
                c->on_disconnect(event_copy.id);
                return get_next_event(event_channel, event_copy, c, ec);
            }

            return true;
        }
        else
        {
            int verrno = errno;
            if(verrno == EBADF) return false;
            boost::system::error_code err(verrno, boost::system::system_category());
            HPX_IBVERBS_THROWS_IF(
                ec
              , err
            );

            return false;
        }
        BOOST_ASSERT(false);
        return false;
    }

    inline void set_nonblocking(int fd, boost::system::error_code &ec)
    {
        int flags = fcntl(fd, F_GETFL);
        int rc = fcntl(fd, F_SETFL, flags | O_NONBLOCK);
        if(rc < 0)
        {
            int verrno = errno;
            boost::system::error_code err(verrno, boost::system::system_category());
            HPX_IBVERBS_THROWS_IF(
                ec
              , err
            );
        }
    }

}}}

#endif
