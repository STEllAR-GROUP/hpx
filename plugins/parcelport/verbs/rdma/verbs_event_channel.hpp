//  Copyright (c) 2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_VERBS_event_channel_HPP
#define HPX_PARCELSET_POLICIES_VERBS_event_channel_HPP

// config
#include <hpx/config/defines.hpp>
//
#include <hpx/lcos/local/mutex.hpp>
#include <hpx/lcos/local/spinlock.hpp>
//
#include <plugins/parcelport/verbs/rdma/rdma_logging.hpp>
#include <plugins/parcelport/verbs/rdma/rdma_locks.hpp>
//
#include <memory>
#include <string>
#include <utility>
//
#include <rdma/rdma_cma.h>
#include <poll.h>
//
namespace hpx {
namespace parcelset {
namespace policies {
namespace verbs
{

    struct verbs_event_channel {
        //
        typedef hpx::lcos::local::spinlock   mutex_type;
        typedef hpx::parcelset::policies::verbs::unique_lock<mutex_type> unique_lock;

        // ----------------------------------------------------------------------------
        enum event_ack_type {
            do_ack_event = true,
            no_ack_event = false
        };

        // ----------------------------------------------------------------------------
        verbs_event_channel() {}

        // ----------------------------------------------------------------------------
        ~verbs_event_channel() {
            // Destroy the event channel.
            if (event_channel_ != nullptr) {
                LOG_TRACE_MSG("destroying rdma event channel with fd "
                    << hexnumber(event_channel_->fd));
                rdma_destroy_event_channel(event_channel_.get()); // No return code
            }
            event_channel_ = nullptr;
        }

        // ----------------------------------------------------------------------------
        bool create_channel()
        {
            auto t = rdma_create_event_channel();
            event_channel_ = std::unique_ptr<struct rdma_event_channel>(t);
            if (event_channel_ == nullptr) {
                rdma_error e(EINVAL, "rdma_create_verbs_event_channel() failed");
                throw e;
            }
            LOG_DEBUG_MSG("created rdma event channel with fd "
                << hexnumber(event_channel_->fd));
            return true;
        }

        // ----------------------------------------------------------------------------
        template<typename Func>
        int poll_verbs_event_channel(Func &&f)
        {
            return poll_verbs_event_channel(get_file_descriptor(), std::forward<Func>(f));
        }

        template<typename Func>
        static int poll_verbs_event_channel(int fd, Func &&f)
        {
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
                // an interrupt is ok, not an epic fail
                if (err == EINTR) {
                    LOG_TRACE_MSG("poll returned EINTR, continuing ...");
                    return 0;
                }
                rdma_error e(err, "poll_verbs_event_channel failed");
                throw e;
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
        int get_event(event_ack_type ack,
            rdma_cm_event_type event, struct rdma_cm_event *&cm_event)
        {
            return get_event(event_channel_.get(), ack, event, cm_event);
        }

        static int get_event(
            struct rdma_event_channel *channel,
            event_ack_type ack,
            rdma_cm_event_type event,
            struct rdma_cm_event *&cm_event)
        {
            cm_event = nullptr;
            // This operation can block if there are no pending events available.
            // (So only call it after the event poll says there is an event waiting)
            LOG_DEVEL_MSG("waiting for " << rdma_event_str(event)
                << " on event channel " << hexnumber(channel->fd));
            int rc = rdma_get_cm_event(channel, &cm_event);
            if (rc != 0) {
                int err = errno;
                LOG_ERROR_MSG("error getting rdma cm event: "
                    << rdma_error::error_string(err));
                return -1;
            }
            LOG_DEVEL_MSG("got " << rdma_event_str(cm_event->event)
                << " on event channel " << hexnumber(channel->fd));

            // we have to ack events, even when they are not the ones we wanted
            if (cm_event->event != event && event!=rdma_cm_event_type(-1)) {
                LOG_ERROR_MSG("mismatch " << rdma_event_str(cm_event->event)
                    << " not " << rdma_event_str(event));
                if (ack) ack_event(cm_event);
                return -1;
            }
            // Acknowledge the event.
            if (ack) return ack_event(cm_event);
            else return 0;
        }

        // ----------------------------------------------------------------------------
        static int ack_event(struct rdma_cm_event *cm_event)
        {
            if (cm_event == nullptr) {
                LOG_ERROR_MSG("nullptr rdma cm event : cannot ack");
                return ENOENT;
            }

            LOG_TRACE_MSG("ack rdma cm event " << rdma_event_str(cm_event->event)
                << " for rdma cm id " << cm_event->id);

            int err = rdma_ack_cm_event(cm_event);
            if (err != 0) {
                rdma_error e(errno, std::string("Failed acking event ") +
                    rdma_event_str(cm_event->event));
                throw e;
            }
            return 0;
        }

        int get_file_descriptor(void) const { return event_channel_->fd; }

        struct rdma_event_channel *get_verbs_event_channel(void) const {
            return event_channel_.get();
        }


    private:
        // Event channel for notification of RDMA connection management events.
        std::unique_ptr<struct rdma_event_channel> event_channel_;
    };

}}}}

#endif
