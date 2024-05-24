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

        explicit constexpr receiver(Parcelport& pp) noexcept
          : pp_(pp), rcv_header_(hpx::util::openshmem_environment::size()), headers_mtx_{}, handles_header_mtx_{},
            handles_header_{}, connections_mtx_{}, connections_{}
        {
        }

        void run() noexcept
        {
            util::openshmem_environment::scoped_lock l;
            post_new_header(l);
        }

        bool background_work() noexcept
        {
            // We first try to accept a new connection
            connection_ptr connection = accept();

            // If we don't have a new connection, try to handle one of the
            // already accepted ones.
            if (!connection)
            {
                std::unique_lock<hpx::spinlock> l(connections_mtx_, std::try_to_lock);
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
                std::unique_lock<hpx::spinlock> l(connections_mtx_);
                connections_.push_back(HPX_MOVE(connection));
            }
        }

        connection_ptr accept() noexcept
        {
            std::unique_lock<hpx::spinlock> l(headers_mtx_, std::try_to_lock);
            if (l.owns_lock())
            {
                return accept_locked(l);
            }
            return connection_ptr();
        }

        template <typename Lock>
        connection_ptr accept_locked(Lock& header_lock)
        {
            HPX_ASSERT_OWNS_LOCK(header_lock);

            util::openshmem_environment::scoped_try_lock l;

            // Caller failing to hold lock 'header_lock' before calling function
#if defined(HPX_MSVC)
#pragma warning(push)
#pragma warning(disable : 26110)
#endif

            if (l.locked)
            {
                const auto idx = post_new_header(l);
                header h = rcv_header_[idx];
                rcv_header_[idx].reset();

                l.unlock();
                header_lock.unlock();

                return std::make_shared<connection_type>(
                    idx, HPX_MOVE(h), pp_);
            }

#if defined(HPX_MSVC)
#pragma warning(pop)
#endif

            return {};
        }

        template <typename Lock>
        std::size_t post_new_header([[maybe_unused]] Lock& l) noexcept
        {
            HPX_ASSERT_OWNS_LOCK(l);

            const auto self_ = hpx::util::openshmem_environment::rank();

            const std::size_t sys_pgsz = sysconf(_SC_PAGESIZE);
            const std::size_t page_count =
                hpx::util::openshmem_environment::size();
            const std::size_t beg_rcv_signal = (sys_pgsz*page_count);

	    int idx = 0;
            for(idx = 0; i < page_count; ++i) {
                if(shmem_test(hpx::util::openshmem_environment::shmem_buffer + beg_rcv_signal + i, SHMEM_CMP_EQ, 1)) {
                    (*(hpx::util::openshmem_environment::segments[idx].rcv)) = 0;
		}
            }

            std::memcpy(reinterpret_cast<std::uint8_t*>(rcv_header_[idx].data()),
                hpx::util::openshmem_environment::segments[idx].beg_addr,
                sizeof(header)
            );		

            HPX_ASSERT_LOCKED(l, idx < 0);
            return idx;
        }

        Parcelport& pp_;
        std::vector<header> rcv_header_;

        hpx::spinlock headers_mtx_;

        hpx::spinlock handles_header_mtx_;
        handles_header_type handles_header_;

        hpx::spinlock connections_mtx_;
        connection_list connections_;
    };
}    // namespace hpx::parcelset::policies::openshmem

#endif
