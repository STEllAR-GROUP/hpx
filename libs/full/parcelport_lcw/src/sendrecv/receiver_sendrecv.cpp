//  Copyright (c) 2025 Jiakun Yan
//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Thomas Heller
//  Copyright (c)      2020 Google
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCW)

#include <hpx/modules/lcw_base.hpp>
#include <hpx/parcelport_lcw/locality.hpp>
#include <hpx/parcelport_lcw/receiver_base.hpp>
#include <hpx/parcelport_lcw/sendrecv/receiver_connection_sendrecv.hpp>
#include <hpx/parcelport_lcw/sendrecv/receiver_sendrecv.hpp>

#include <hpx/assert.hpp>
#include <cstddef>
#include <memory>

namespace hpx::parcelset::policies::lcw {
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

        auto poll_comp_start = util::lcw_environment::pcounter_now();
        auto completion_manager_p = pp_->get_tls_device().completion_manager_p;
        ::lcw::request_t request;
        bool poll_ret = completion_manager_p->recv_new->poll(request);
        util::lcw_environment::pcounter_add(util::lcw_environment::poll_comp,
            util::lcw_environment::pcounter_since(poll_comp_start));

        if (poll_ret)
        {
            auto useful_bg_start = util::lcw_environment::pcounter_now();
            util::lcw_environment::log(
                util::lcw_environment::log_level_t::debug, "recv",
                "accept_new (%d, %d, %d) length %lu\n", request.rank,
                ::lcw::get_rank(), request.tag, request.length);
            connection_ptr connection = create_connection(request.rank, pp_);
            connection->load((char*) request.buffer);
            free(request.buffer);
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
            util::lcw_environment::pcounter_add(
                util::lcw_environment::useful_bg_work,
                util::lcw_environment::pcounter_since(useful_bg_start));
            did_some_work = true;
        }
        return did_some_work;
    }

    bool receiver_sendrecv::followup() noexcept
    {
        bool did_some_work = false;
        // We don't use a request_wrapper here because all the receive buffers
        // should be managed by the connections
        auto poll_comp_start = util::lcw_environment::pcounter_now();
        auto completion_manager_p = pp_->get_tls_device().completion_manager_p;
        ::lcw::request_t request;
        bool poll_ret = completion_manager_p->recv_followup->poll(request);
        util::lcw_environment::pcounter_add(util::lcw_environment::poll_comp,
            util::lcw_environment::pcounter_since(poll_comp_start));

        if (poll_ret)
        {
            auto useful_bg_start = util::lcw_environment::pcounter_now();
            HPX_ASSERT(request.user_context);
            auto* sharedPtr_p = (connection_ptr*) request.user_context;
            util::lcw_environment::log(
                util::lcw_environment::log_level_t::debug, "recv",
                "followup (%d, %d, %d) length %lu\n", request.rank,
                ::lcw::get_rank(), request.tag, request.length);
            receiver_connection_sendrecv::return_t ret =
                (*sharedPtr_p)->receive();
            if (ret.isDone)
            {
                (*sharedPtr_p)->done();
                delete sharedPtr_p;
            }
            else
            {
                completion_manager_p->recv_followup->enqueue_completion(
                    ret.completion);
            }
            util::lcw_environment::pcounter_add(
                util::lcw_environment::useful_bg_work,
                util::lcw_environment::pcounter_since(useful_bg_start));
            did_some_work = true;
        }
        return did_some_work;
    }
}    // namespace hpx::parcelset::policies::lcw

#endif
