//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c) 2014-2015 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_MPI)
#include <hpx/assert.hpp>
#include <hpx/modules/mpi_base.hpp>

#include <hpx/parcelport_mpi/header.hpp>
#include <hpx/parcelport_mpi/receiver_connection.hpp>

#include <algorithm>
#include <deque>
#include <iterator>
#include <list>
#include <memory>
#include <mutex>
#include <set>
#include <utility>

#include <mpi.h>

namespace hpx::parcelset::policies::mpi {

    template <typename Parcelport>
    struct receiver
    {
        using header_list = std::list<std::pair<int, header>>;
        using handles_header_type = std::set<std::pair<int, int>>;
        using connection_type = receiver_connection<Parcelport>;
        using connection_ptr = std::shared_ptr<connection_type>;
        using connection_list = std::deque<connection_ptr>;

        explicit constexpr receiver(Parcelport& pp) noexcept
          : pp_(pp)
          , hdr_request_(0)
        {
        }

        void run() noexcept
        {
            util::mpi_environment::scoped_lock l;
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
        connection_ptr accept_locked(Lock& header_lock)
        {
            connection_ptr res;
            util::mpi_environment::scoped_try_lock l;

            // Caller failing to hold lock 'header_lock' before calling function
#if defined(HPX_MSVC)
#pragma warning(push)
#pragma warning(disable : 26110)
#endif

            if (l.locked)
            {
                MPI_Status status;
                if (request_done_locked(hdr_request_, &status))
                {
                    header h = new_header();
                    l.unlock();
                    header_lock.unlock();

                    res.reset(new connection_type(status.MPI_SOURCE, h, pp_));
                    return res;
                }
            }

#if defined(HPX_MSVC)
#pragma warning(pop)
#endif

            return res;    //-V614
        }

        header new_header() noexcept
        {
            header h = rcv_header_;
            rcv_header_.reset();

            MPI_Irecv(rcv_header_.data(), rcv_header_.data_size_, MPI_BYTE,
                MPI_ANY_SOURCE, 0, util::mpi_environment::communicator(),
                &hdr_request_);

            return h;
        }

        Parcelport& pp_;

        hpx::spinlock headers_mtx_;
        MPI_Request hdr_request_;
        header rcv_header_;

        hpx::spinlock handles_header_mtx_;
        handles_header_type handles_header_;

        hpx::spinlock connections_mtx_;
        connection_list connections_;

        bool request_done_locked(MPI_Request& r, MPI_Status* status) noexcept
        {
            int completed = 0;
            int ret = MPI_Test(&r, &completed, status);
            HPX_ASSERT(ret == MPI_SUCCESS);
            (void) ret;
            if (completed)
            {
                return true;
            }
            return false;
        }
    };
}    // namespace hpx::parcelset::policies::mpi

#endif
