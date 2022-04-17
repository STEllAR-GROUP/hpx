
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
#include <hpx/parcelport_lci/header.hpp>
#include <hpx/parcelport_lci/receiver_connection.hpp>

#include <algorithm>
#include <deque>
#include <iterator>
#include <list>
#include <memory>
#include <mutex>
#include <set>
#include <utility>

namespace hpx::parcelset::policies::lci {
    template <typename Parcelport>
    struct receiver
    {
        using mutex_type = hpx::spinlock;
        using header_list = std::list<std::pair<int, header>>;
        using handles_header_type = std::set<std::pair<int, int>>;
        using connection_type = receiver_connection<Parcelport>;
        using connection_ptr = std::shared_ptr<connection_type>;
        using connection_list = std::deque<connection_ptr>;

        explicit receiver(Parcelport& pp) noexcept
          : pp_(pp)
        {
        }

        void run() noexcept {}

        bool background_work() noexcept
        {
            // We first try to accept a new connection
            connection_ptr connection = accept();

            // If we don't have a new connection, try to handle one of the
            // already accepted ones.
            if (!connection)
            {
                std::unique_lock<mutex_type> l(
                    connections_mtx_, std::try_to_lock);
                if (l.owns_lock() && !connections_.empty())
                {
                    connection = HPX_MOVE(connections_.front());
                    connections_.pop_front();
                }
            }

            if (connection)
            {
                receive_messages(HPX_MOVE(connection));
                return true;
            }

            return false;
        }

        void receive_messages(connection_ptr connection) noexcept
        {
            if (!connection->receive())
            {
                std::unique_lock<mutex_type> l(connections_mtx_);
                connections_.push_back(HPX_MOVE(connection));
            }
        }

        connection_ptr accept() noexcept
        {
            connection_ptr res;
            LCI_request_t request;
            LCI_error_t ret =
                LCI_queue_pop(util::lci_environment::h_queue(), &request);
            if (ret == LCI_OK)
            {
                header h = *(header*) (request.data.mbuffer.address);
                h.assert_valid();

                res.reset(new connection_type(request.rank, h, pp_));
                LCI_mbuffer_free(request.data.mbuffer);
            }
            return res;
        }

        Parcelport& pp_;

        mutex_type headers_mtx_;
        header rcv_header_;

        mutex_type handles_header_mtx_;
        handles_header_type handles_header_;

        mutex_type connections_mtx_;
        connection_list connections_;
    };

}    // namespace hpx::parcelset::policies::lci

#endif
