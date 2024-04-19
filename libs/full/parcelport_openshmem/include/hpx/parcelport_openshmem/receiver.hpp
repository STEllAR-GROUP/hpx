//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c) 2014-2015 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_OPENSHMEM)
#include <hpx/assert.hpp>
#include <hpx/modules/openshmem_base.hpp>

#include <hpx/parcelport_openshmem/header.hpp>
#include <hpx/parcelport_openshmem/receiver_connection.hpp>

#include <algorithm>
#include <chrono>
#include <deque>
#include <iterator>
#include <list>
#include <memory>
#include <mutex>
#include <set>
#include <utility>

namespace hpx::parcelset::policies::openshmem {

    template <typename Parcelport>
    struct receiver
    {
        using header_list = std::list<std::pair<int, header>>;
        using handles_header_type = std::set<std::pair<int, int>>;
        using connection_type = receiver_connection<Parcelport>;
        using connection_ptr = std::shared_ptr<connection_type>;
        using connection_list = std::deque<connection_ptr>;

/*
        struct exp_backoff
        {
            int numTries;
            const static int maxRetries = 10;

            void operator()(unsigned int* addr)
            {
                if (numTries <= maxRetries)
                {
                    if (shmem_uint_test(addr, SHMEM_CMP_EQ, 1))
                    {
                        return;
                    }
                    hpx::this_thread::suspend(
                        std::chrono::microseconds(1 << numTries));
                }
                else
                {
                    numTries = 0;
                }
            }
        };
*/

        explicit constexpr receiver(Parcelport& pp) noexcept
          : pp_(pp)
          //, bo()
        {
        }

        void run() noexcept
        {
            //util::openshmem_environment::scoped_lock l;
            //new_header(-1);
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
            util::openshmem_environment::scoped_try_lock l;

            if (l.locked)
            {
                header h = new_header(res->sending_thd_id_);
                l.unlock();
                header_lock.unlock();

                // remote localities 'put' into the openshmem shared
                // memory segment on this machine
                //
                res.reset(new connection_type(
                    hpx::util::openshmem_environment::rank(), h, pp_));
                return res;
            }
            return res;
        }

        header new_header(const auto sending_thd_id) noexcept
        {
            const auto self_ = hpx::util::openshmem_environment::rank();
            const auto nthreads_ = hpx::util::openshmem_environment::nthreads_;

            const std::size_t sys_pgsz = sysconf(_SC_PAGESIZE);
            const std::size_t page_count =
                hpx::util::openshmem_environment::size() * hpx::util::openshmem_environment::nthreads_;
            const std::size_t beg_rcv_signal = (sys_pgsz*page_count);

            // waiting for `sender_connection::send_header` invocation
            while (rcv_header_.data() == 0) {

                const auto idx = hpx::util::openshmem_environment::wait_until_any(
                    1,
                    reinterpret_cast<unsigned int *>(hpx::util::openshmem_environment::shmem_buffer + beg_rcv_signal),
                    page_count
                );

                hpx::util::openshmem_environment::get(
                    reinterpret_cast<std::uint8_t*>(
                       rcv_header_.data()),
                    self_,
                    hpx::util::openshmem_environment::segments[idx].beg_addr,
                    sizeof(rcv_header_)
                );
            }

            header h = rcv_header_;
            rcv_header_.reset();
            return h;
        }

        Parcelport& pp_;

        hpx::spinlock headers_mtx_;
        header rcv_header_;

        hpx::spinlock handles_header_mtx_;
        handles_header_type handles_header_;

        hpx::spinlock connections_mtx_;
        connection_list connections_;
        //exp_backoff bo;
    };
}    // namespace hpx::parcelset::policies::openshmem

#endif
