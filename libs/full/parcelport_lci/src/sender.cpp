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
#include <hpx/parcelport_lci/backlog_queue.hpp>
#include <hpx/parcelport_lci/locality.hpp>
#include <hpx/parcelport_lci/receiver.hpp>
#include <hpx/parcelport_lci/sender.hpp>
#include <hpx/parcelport_lci/sender_connection.hpp>

#include <memory>

namespace hpx::parcelset::policies::lci {
    sender::connection_ptr sender::create_connection(
        int dest, parcelset::parcelport* pp)
    {
        return std::make_shared<connection_type>(dest, pp);
    }

    bool sender::background_work(size_t /* num_thread */) noexcept
    {
        bool did_some_work = false;
        // try to accept a new connection
        LCI_request_t request;
        request.flag = LCI_ERR_RETRY;
        LCI_queue_pop(util::lci_environment::get_scq(), &request);

        if (request.flag == LCI_OK)
        {
            auto* sharedPtr_p = (connection_ptr*) request.user_context;
            (*sharedPtr_p)->done();
            delete sharedPtr_p;
            did_some_work = true;
        }

        return did_some_work;
    }

    bool sender::send(parcelset::parcelport* pp,
        parcelset::locality const& dest, parcel_buffer_type buffer,
        callback_fn_type&& callbackFn)
    {
        int dest_rank = dest.get<locality>().rank();
        auto connection = std::make_shared<sender_connection>(dest_rank, pp);
        connection->buffer_ = HPX_MOVE(buffer);
        connection->async_write(HPX_MOVE(callbackFn), nullptr);
        return true;
    }

}    // namespace hpx::parcelset::policies::lci

#endif
