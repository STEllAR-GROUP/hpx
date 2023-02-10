//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCI)

#include <hpx/assert.hpp>
#include <hpx/synchronization/spinlock.hpp>

#include <hpx/modules/lci_base.hpp>
#include <hpx/parcelport_lci/sender_connection.hpp>

#include <algorithm>
#include <iterator>
#include <list>
#include <memory>
#include <mutex>
#include <utility>

namespace hpx::parcelset::policies::lci {
    struct sender
    {
        using connection_type = sender_connection;
        using connection_ptr = std::shared_ptr<connection_type>;

        sender() noexcept {}

        void run() noexcept {}

        connection_ptr create_connection(int dest, parcelset::parcelport* pp)
        {
            return std::make_shared<connection_type>(this, dest, pp);
        }

        bool background_work() noexcept
        {
            // We first try to accept a new connection
            LCI_request_t request;
            request.flag = LCI_ERR_RETRY;
            LCI_queue_pop(util::lci_environment::get_scq(), &request);

            if (request.flag == LCI_OK)
            {
                auto* sharedPtr_p = (connection_ptr*) request.user_context;
                (*sharedPtr_p)->done();
                delete sharedPtr_p;
                return true;
            }
            return false;
        }
    };

}    // namespace hpx::parcelset::policies::lci

#endif
