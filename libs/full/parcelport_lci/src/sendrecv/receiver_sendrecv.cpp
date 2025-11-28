//  Copyright (c) 2023-2024 Jiakun Yan
//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Thomas Heller
//  Copyright (c)      2020 Google
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCI)

#include <hpx/modules/lci_base.hpp>
#include <hpx/parcelport_lci/locality.hpp>
#include <hpx/parcelport_lci/receiver_base.hpp>
#include <hpx/parcelport_lci/sendrecv/receiver_connection_sendrecv.hpp>
#include <hpx/parcelport_lci/sendrecv/receiver_sendrecv.hpp>

#include <hpx/assert.hpp>
#include <cstddef>
#include <memory>

namespace hpx::parcelset::policies::lci {
    receiver_sendrecv::connection_ptr receiver_sendrecv::create_connection(
        int dest, parcelset::parcelport* pp)
    {
        return std::make_shared<receiver_connection_sendrecv>(dest, pp);
    }

    bool receiver_sendrecv::background_work() noexcept
    {
        bool did_something = false;
        // We first try to accept a new connection
        did_something = accept_new() || did_something;
        // We will then try to make progress on existing connections
        did_something = followup() || did_something;
        return did_something;
    }

    bool receiver_sendrecv::accept_new() noexcept
    {
        bool did_some_work = false;

        auto poll_comp_start = util::lci_environment::pcounter_now();
        auto completion_manager_p = pp_->get_tls_device().completion_manager_p;
        auto status = completion_manager_p->recv_new->poll();
        util::lci_environment::pcounter_add(util::lci_environment::poll_comp,
            util::lci_environment::pcounter_since(poll_comp_start));

        if (status.is_done())
        {
            auto useful_bg_start = util::lci_environment::pcounter_now();
            if (config_t::protocol == config_t::protocol_t::sendrecv)
            {
                HPX_ASSERT(false);
                // std::size_t device_idx =
                //     (std::size_t) status.get_user_context();
                // auto& device = pp_->devices[device_idx];
                // ::lci::comp_t completion =
                //     completion_manager_p->recv_new->alloc_completion();
                // LCI_recvmn(device.endpoint_new, util::lci_environment::rank()_ANY,
                //     0, completion, reinterpret_cast<void*>(device_idx));
                // completion_manager_p->recv_new->enqueue_completion(completion);
            }
            util::lci_environment::log(
                util::lci_environment::log_level_t::debug, "recv",
                "accept_new (%d, %d, %d) length %lu\n", status.get_rank(),
                util::lci_environment::rank(), status.get_tag(),
                status.get_size());
            connection_ptr connection =
                create_connection(status.get_rank(), pp_);
            connection->load((char*) status.get_buffer());
            if (config_t::protocol == config_t::protocol_t::putsendrecv)
            {
                ::lci::put_upacket(status.get_buffer());
            }
            receiver_connection_sendrecv::return_t ret = connection->receive();
            if (ret.isDone)
            {
                connection->done();
            }
            else
            {
                completion_manager_p->recv_followup->enqueue_completion(
                    ret.completion);
            }
            util::lci_environment::pcounter_add(
                util::lci_environment::useful_bg_work,
                util::lci_environment::pcounter_since(useful_bg_start));
            did_some_work = true;
        }
        return did_some_work;
    }

    bool receiver_sendrecv::followup() noexcept
    {
        bool did_some_work = false;
        // We don't use a request_wrapper here because all the receive buffers
        // should be managed by the connections
        auto poll_comp_start = util::lci_environment::pcounter_now();
        auto completion_manager_p = pp_->get_tls_device().completion_manager_p;
        ::lci::status_t status = completion_manager_p->recv_followup->poll();
        util::lci_environment::pcounter_add(util::lci_environment::poll_comp,
            util::lci_environment::pcounter_since(poll_comp_start));

        if (status.is_done())
        {
            auto useful_bg_start = util::lci_environment::pcounter_now();
            HPX_ASSERT(status.get_user_context());
            auto* sharedPtr_p = (connection_ptr*) status.get_user_context();
            size_t length = status.get_size();
            util::lci_environment::log(
                util::lci_environment::log_level_t::debug, "recv",
                "followup (%d, %d, %d) length %lu\n", status.get_rank(),
                util::lci_environment::rank(), status.get_tag(), length);
            receiver_connection_sendrecv::return_t ret =
                (*sharedPtr_p)->receive();
            if (ret.isDone)
            {
                (*sharedPtr_p)->done();
            }
            else
            {
                completion_manager_p->recv_followup->enqueue_completion(
                    ret.completion);
            }
            util::lci_environment::pcounter_add(
                util::lci_environment::useful_bg_work,
                util::lci_environment::pcounter_since(useful_bg_start));
            did_some_work = true;
        }
        return did_some_work;
    }
}    // namespace hpx::parcelset::policies::lci

#endif
