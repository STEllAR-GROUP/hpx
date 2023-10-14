//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c) 2014-2015 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_GASNET)
#include <hpx/assert.hpp>
#include <hpx/modules/gasnet_base.hpp>

#include <hpx/parcelport_gasnet/header.hpp>
#include <hpx/parcelport_gasnet/receiver_connection.hpp>

#include <algorithm>
#include <chrono>
#include <deque>
#include <iterator>
#include <list>
#include <memory>
#include <mutex>
#include <set>
#include <utility>

namespace hpx::parcelset::policies::gasnet {

    template <typename Parcelport>
    struct receiver
    {
        using header_list = std::list<std::pair<int, header>>;
        using handles_header_type = std::set<std::pair<int, int>>;
        using connection_type = receiver_connection<Parcelport>;
        using connection_ptr = std::shared_ptr<connection_type>;
        using connection_list = std::deque<connection_ptr>;

        struct exp_backoff
        {
            int numTries;
            const static int maxRetries = 10;

            void operator()()
            {
                if (numTries <= maxRetries)
                {
                    gasnet_AMPoll();
                    hpx::this_thread::suspend(
                        std::chrono::microseconds(1 << numTries));
                }
                else
                {
                    numTries = 0;
                }
            }
        };

        explicit constexpr receiver(Parcelport& pp) noexcept
          : pp_(pp)
          , bo()
        {
        }

        void run() noexcept
        {
            util::gasnet_environment::scoped_lock l;
            new_header();
        }

        bool background_work() noexcept
        {
            // We first try to accept a new connection
            connection_ptr connection = accept();

            // If we don't have a new connection, try to handle one of the
            // already accepted ones.
            if (!connection)
            {
                std::unique_lock l(connections_mtx_, std::try_to_lock);
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
                std::unique_lock l(connections_mtx_);
                connections_.push_back(HPX_MOVE(connection));
            }
        }

        connection_ptr accept() noexcept
        {
            std::unique_lock l(headers_mtx_, std::try_to_lock);
            if (l.owns_lock())
            {
                return accept_locked(l);
            }
            return connection_ptr();
        }

        template <typename Lock>
        connection_ptr accept_locked(Lock& header_lock) noexcept
        {
            connection_ptr res;
            util::gasnet_environment::scoped_try_lock l;

            if (l.locked)
            {
                header h = new_header();
                l.unlock();
                header_lock.unlock();

                // remote localities 'put' into the gasnet shared
                // memory segment on this machine
                //
                res.reset(new connection_type(
                    hpx::util::gasnet_environment::rank(), h, pp_));
                return res;
            }
            return res;
        }

        header new_header() noexcept
        {
            header h = rcv_header_;
            rcv_header_.reset();

            while (rcv_header_.data() == 0)
            {
                bo();
            }

            return h;
        }

        Parcelport& pp_;

        hpx::spinlock headers_mtx_;
        header rcv_header_;

        hpx::spinlock handles_header_mtx_;
        handles_header_type handles_header_;

        hpx::spinlock connections_mtx_;
        connection_list connections_;
        exp_backoff bo;
    };

}    // namespace hpx::parcelset::policies::gasnet

#endif
