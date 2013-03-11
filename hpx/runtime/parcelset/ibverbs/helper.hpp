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
    inline bool get_next_event(rdma_event_channel *event_channel, rdma_cm_event & event_copy, Connection * c)
    {
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
            // FIXME: error
            return false;
        }

        if(rdma_get_cm_event(event_channel, &event) == 0)
        {
            std::memcpy(&event_copy, event, sizeof(rdma_cm_event));

            rdma_ack_cm_event(event);

            if(event_copy.event == RDMA_CM_EVENT_DISCONNECTED)
            {
                c->on_disconnect(event_copy.id);
                return get_next_event(event_channel, event_copy, c);
            }

            return true;
        }
        else
        {
            int err = errno;
            if(err == EBADF) return false;

            //FIXME: error

            return false;
        }
        BOOST_ASSERT(false);
        return false;
    }

    inline void set_nonblocking(int fd)
    {
        int flags = fcntl(fd, F_GETFL);
        int rc = fcntl(fd, F_SETFL, flags | O_NONBLOCK);
        if(rc < 0)
        {
            //FIXME: error:
        }
    }

}}}

#endif
