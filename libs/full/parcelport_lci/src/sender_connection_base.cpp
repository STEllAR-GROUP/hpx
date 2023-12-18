//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Thomas Heller
//  Copyright (c)      2020 Google
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCI)

#include <hpx/assert.hpp>

#include <hpx/modules/lci_base.hpp>
#include <hpx/parcelport_lci/backlog_queue.hpp>
#include <hpx/parcelport_lci/header.hpp>
#include <hpx/parcelport_lci/locality.hpp>
#include <hpx/parcelport_lci/parcelport_lci.hpp>
#include <hpx/parcelport_lci/receiver_base.hpp>
#include <hpx/parcelport_lci/sender_connection_base.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace hpx::parcelset::policies::lci {
    void sender_connection_base::async_write(
        sender_connection_base::handler_type&& handler,
        sender_connection_base::postprocess_handler_type&& parcel_postprocess)
    {
        LCT_time_t async_write_start_time =
            util::lci_environment::pcounter_now();
        device_p = &pp_->get_tls_device();
        load(HPX_FORWARD(handler_type, handler),
            HPX_FORWARD(postprocess_handler_type, parcel_postprocess));
        return_t ret = send();
        if (ret.status == return_status_t::done)
        {
            done();
        }
        else if (ret.status == return_status_t::wait)
        {
            device_p->completion_manager_p->send->enqueue_completion(
                ret.completion);
        }
        util::lci_environment::pcounter_add(
            util::lci_environment::async_write_timer,
            util::lci_environment::pcounter_since(async_write_start_time));
    }

    sender_connection_base::return_t sender_connection_base::send()
    {
        auto start_time = util::lci_environment::pcounter_now();
        return_t ret;
        const int retry_max_spin = 32;
        if (!config_t::enable_lci_backlog_queue ||
            HPX_UNLIKELY(!pp_->is_initialized))
        {
            // If we are sending early parcels, we should not expect the
            // thread make progress on the backlog queue
            int retry_count = 0;
            do
            {
                ret = send_nb();
                if (ret.status == return_status_t::retry)
                {
                    ++retry_count;
                    if (retry_count > retry_max_spin)
                    {
                        retry_count = 0;
                        while (pp_->background_work(
                            -1, parcelport_background_mode_all))
                            continue;
                        if (hpx::threads::get_self_id() !=
                            hpx::threads::invalid_thread_id)
                            hpx::this_thread::yield();
                    }
                    if (config_t::progress_type ==
                            config_t::progress_type_t::worker ||
                        config_t::progress_type ==
                            config_t::progress_type_t::pthread_worker)
                        while (pp_->do_progress_local())
                            continue;
                }
            } while (ret.status == return_status_t::retry);
        }
        else
        {
            if (!backlog_queue::empty(dst_rank))
            {
                backlog_queue::push(shared_from_this());
                ret = {return_status_t::retry, nullptr};
            }
            else
            {
                ret = send_nb();
                if (ret.status == return_status_t::retry)
                {
                    backlog_queue::push(shared_from_this());
                }
            }
        }
        util::lci_environment::pcounter_add(util::lci_environment::send_timer,
            util::lci_environment::pcounter_since(start_time));
        return ret;
    }

    void sender_connection_base::profile_start_hook(const header& header_)
    {
        util::lci_environment::pcounter_add(
            util::lci_environment::send_conn_start, 1);
        if (util::lci_environment::log_level <
            util::lci_environment::log_level_t::profile)
            return;
        char buf[1024];
        size_t consumed = 0;
        consumed += snprintf(buf + consumed, sizeof(buf) - consumed,
            "%d:%lf:send_connection(%p) start:%d:%d:%d:[", LCI_RANK,
            hpx::chrono::high_resolution_clock::now() / 1e9, (void*) this,
            header_.numbytes_nonzero_copy(), header_.numbytes_tchunk(),
            header_.num_zero_copy_chunks());
        HPX_ASSERT(sizeof(buf) > consumed);
        for (int i = 0; i < header_.num_zero_copy_chunks(); ++i)
        {
            std::string format = "%lu,";
            if (i == header_.num_zero_copy_chunks() - 1)
                format = "%lu";
            consumed += snprintf(buf + consumed, sizeof(buf) - consumed,
                format.c_str(), buffer_.transmission_chunks_[i].second);
            HPX_ASSERT(sizeof(buf) > consumed);
        }
        consumed += snprintf(buf + consumed, sizeof(buf) - consumed, "]\n");
        HPX_ASSERT(sizeof(buf) > consumed);
        util::lci_environment::log(
            util::lci_environment::log_level_t::profile, "send", "%s", buf);
    }

    void sender_connection_base::profile_end_hook()
    {
        util::lci_environment::log(util::lci_environment::log_level_t::profile,
            "send", "%d:%lf:send_connection(%p) end\n", LCI_RANK,
            hpx::chrono::high_resolution_clock::now() / 1e9, (void*) this);
        util::lci_environment::pcounter_add(
            util::lci_environment::send_conn_end, 1);
    }
}    // namespace hpx::parcelset::policies::lci

#endif
