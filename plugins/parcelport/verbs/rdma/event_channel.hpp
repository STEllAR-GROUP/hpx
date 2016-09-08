//  Copyright (c) 2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_VERBS_EVENT_CHANNEL_HPP
#define HPX_PARCELSET_POLICIES_VERBS_EVENT_CHANNEL_HPP

// config
#include <hpx/config/defines.hpp>
//
#include <hpx/lcos/local/mutex.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/lcos/local/condition_variable.hpp>
//
#include <plugins/parcelport/verbs/rdma/rdma_logging.hpp>
//
#include <rdma/rdma_cma.h>
#include <poll.h>
//
namespace hpx {
namespace parcelset {
namespace policies {
namespace verbs {

    struct event_channel {
        //
        typedef hpx::lcos::local::spinlock   mutex_type;
        typedef std::unique_lock<mutex_type> unique_lock;
        //
        static mutex_type event_mutex_;

        // ----------------------------------------------------------------------------
        enum event_ack_type {
            do_ack_event = true,
            no_ack_event = false
        };

        // ----------------------------------------------------------------------------
        template<typename Func>
        static int poll_event_channel(int fd, Func &&f)
        {
            // there is no need for more than one thread to poll the event channel
            // so try the lock and if someone has it, leave immediately
            unique_lock lock(event_mutex_, std::try_to_lock);
            if (!lock.owns_lock()) {
                return 0;
            }

            const int eventChannel = 0;
            const int numFds = 1;
            //
            pollfd pollInfo[numFds];
            int polltimeout = 0; // seconds*1000; // 10000 == 10 sec

            pollInfo[eventChannel].fd = fd;
            pollInfo[eventChannel].events = POLLIN;
            pollInfo[eventChannel].revents = 0;

            // Wait for an event on one of the descriptors.
            int rc = poll(pollInfo, numFds, polltimeout);

            // If there were no events/messages
            if (rc == 0) {
                return 0;
            }

            // There was an error so log the failure and try again.
            if (rc == -1) {
                int err = errno;
                if (err == EINTR) {
                    LOG_TRACE_MSG("poll returned EINTR, continuing ...");
                    return 0;
                }
                LOG_ERROR_MSG(
                    "error polling socket descriptors: " << rdma_error::error_string(err));
                return 0;
            }

            if (pollInfo[eventChannel].revents & POLLIN) {
                LOG_TRACE_MSG("input event available on event channel");
                f();
            }
            return 1;
        }

        // ----------------------------------------------------------------------------
        // If the expected event is received, it is acked, otherwise the
        // event is returned in the event param (but it turns out we must ack
        // even the wrong events so this is done too)
        // Communication event details are returned in the rdma_cm_event structure.
        // It is allocated by the rdma_cm and released by the rdma_ack_cm_event routine.
        static int get_event(struct rdma_event_channel *channel, event_ack_type ack,
            rdma_cm_event_type event, struct rdma_cm_event *&cm_event)
        {
            cm_event = NULL;
            // This operation can block if there are no pending events available.
            LOG_TRACE_MSG("waiting for " << rdma_event_str(event)
                << " on event channel " << hexnumber(channel->fd));
            int rc = rdma_get_cm_event(channel, &cm_event);
            if (rc != 0) {
                int err = errno;
                LOG_ERROR_MSG("error getting rdma cm event: "
                    << rdma_error::error_string(err));
                return -1;
            }

            if (ack) {
                // we have to ack events, even when they are not the ones we wanted
                if (cm_event->event != event) {
                    LOG_ERROR_MSG(" mismatch " << rdma_event_str(cm_event->event)
                        << " not " << rdma_event_str(event));
                    ack_event(cm_event);
                    return -1;
                }
                // Acknowledge the event.
                return ack_event(cm_event);
            }
            return 0;
        }

        // ----------------------------------------------------------------------------
        static int ack_event(struct rdma_cm_event *cm_event)
        {
            if (cm_event == nullptr) {
                LOG_ERROR_MSG("NULL rdma cm event : cannot ack");
                return ENOENT;
            }

            LOG_TRACE_MSG("ack rdma cm event " << rdma_event_str(cm_event->event)
                << " for rdma cm id " << cm_event->id);

            int err = rdma_ack_cm_event(cm_event);
            if (err != 0) {
                err = abs(err);
                LOG_ERROR_MSG("Failed acking event " << rdma_event_str(cm_event->event)
                    << ": " << rdma_error::error_string(err));
                return err;
            }
            return 0;
        }

    };

}}}}

#endif
