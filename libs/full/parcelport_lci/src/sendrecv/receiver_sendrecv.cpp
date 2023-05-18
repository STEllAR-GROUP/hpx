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
        request_wrapper_t request;
        request.request = pp_->recv_new_completion_manager->poll();

        if (request.request.flag == LCI_OK)
        {
            if (config_t::protocol == config_t::protocol_t::sendrecv)
            {
                LCI_comp_t completion =
                    pp_->recv_new_completion_manager->alloc_completion();
                LCI_recvmn(pp_->endpoint_new_eager, LCI_RANK_ANY, 0, completion,
                    nullptr);
                pp_->recv_new_completion_manager->enqueue_completion(
                    completion);
            }
            HPX_ASSERT(request.request.flag == LCI_OK);
            util::lci_environment::log(
                util::lci_environment::log_level_t::debug,
                "accept_new (%d, %d, %d) length %lu\n", request.request.rank,
                LCI_RANK, request.request.tag,
                request.request.data.mbuffer.length);
            connection_ptr connection =
                create_connection(request.request.rank, pp_);
            connection->load((char*) request.request.data.mbuffer.address);
            receiver_connection_sendrecv::return_t ret = connection->receive();
            if (ret.isDone)
            {
                connection->done();
            }
            else
            {
                pp_->recv_followup_completion_manager->enqueue_completion(
                    ret.completion);
            }
            return true;
        }
        return false;
    }

    bool receiver_sendrecv::followup() noexcept
    {
        // We don't use a request_wrapper here because all the receive buffers
        // should be managed by the connections
        LCI_request_t request = pp_->recv_followup_completion_manager->poll();

        if (request.flag == LCI_OK)
        {
            HPX_ASSERT(request.user_context);
            auto* sharedPtr_p = (connection_ptr*) request.user_context;
            size_t length;
            if (request.type == LCI_MEDIUM)
                length = request.data.mbuffer.length;
            else
                length = request.data.lbuffer.length;
            util::lci_environment::log(
                util::lci_environment::log_level_t::debug,
                "followup (%d, %d, %d) length %lu\n", request.rank, LCI_RANK,
                request.tag, length);
            receiver_connection_sendrecv::return_t ret =
                (*sharedPtr_p)->receive();
            if (ret.isDone)
            {
                (*sharedPtr_p)->done();
                delete sharedPtr_p;
            }
            else
            {
                pp_->recv_followup_completion_manager->enqueue_completion(
                    ret.completion);
            }
            return true;
        }
        return false;
    }
}    // namespace hpx::parcelset::policies::lci

#endif
