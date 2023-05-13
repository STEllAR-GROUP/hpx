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
#include <hpx/modules/functional.hpp>
#include <hpx/modules/mpi_base.hpp>
#include <hpx/modules/synchronization.hpp>
#include <hpx/modules/thread_support.hpp>
#include <hpx/parcelport_mpi/sender_connection.hpp>
#include <hpx/parcelport_mpi/tag_provider.hpp>

#include <algorithm>
#include <iterator>
#include <list>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace hpx::parcelset::policies::mpi {

    struct sender
    {
        using connection_type = sender_connection;
        using connection_ptr = std::shared_ptr<connection_type>;
        using connection_list = std::deque<connection_ptr>;

        // different versions of clang-format disagree
        // clang-format off
        sender() noexcept
          : next_free_tag_request_((MPI_Request) (-1))
          , next_free_tag_(-1)
        {
        }
        // clang-format on

        void run() noexcept
        {
            util::mpi_environment::scoped_lock l;
            get_next_free_tag(l);
        }

        connection_ptr create_connection(int dest, parcelset::parcelport* pp)
        {
            return std::make_shared<connection_type>(this, dest, pp);
        }

        void add(connection_ptr const& ptr)
        {
            std::unique_lock l(connections_mtx_);
            connections_.push_back(ptr);
        }

        int acquire_tag() noexcept
        {
            return tag_provider_.acquire();
        }

        void send_messages(connection_ptr connection)
        {
            // Check if sending has been completed....
            if (connection->send())
            {
                error_code const ec(throwmode::lightweight);
                hpx::move_only_function<void(error_code const&,
                    parcelset::locality const&, connection_ptr)>
                    postprocess_handler;
                std::swap(
                    postprocess_handler, connection->postprocess_handler_);
                if (postprocess_handler)
                    postprocess_handler(
                        ec, connection->destination(), connection);
            }
            else
            {
                std::unique_lock l(connections_mtx_);
                connections_.push_back(HPX_MOVE(connection));
            }
        }

        bool background_work() noexcept
        {
            connection_ptr connection;
            {
                std::unique_lock const l(connections_mtx_, std::try_to_lock);
                if (l && !connections_.empty())
                {
                    connection = HPX_MOVE(connections_.front());
                    connections_.pop_front();
                }
            }

            bool has_work = false;
            if (connection)
            {
                send_messages(HPX_MOVE(connection));
                has_work = true;
            }
            next_free_tag();
            return has_work;
        }

        using buffer_type = std::vector<char>;
        using chunk_type = serialization::serialization_chunk;
        using parcel_buffer_type = parcel_buffer<buffer_type, chunk_type>;
        using callback_fn_type =
            hpx::move_only_function<void(error_code const&)>;
        bool send_immediate(parcelset::parcelport* pp,
            parcelset::locality const& dest, parcel_buffer_type buffer,
            callback_fn_type&& callbackFn)
        {
            int dest_rank = dest.get<locality>().rank();
            auto connection = create_connection(dest_rank, pp);
            connection->buffer_ = HPX_MOVE(buffer);
            connection->async_write(HPX_MOVE(callbackFn), nullptr);
            return true;
        }

    private:
        tag_provider tag_provider_;

        void next_free_tag() noexcept
        {
            int next_free = -1;
            {
                std::unique_lock l(next_free_tag_mtx_, std::try_to_lock);
                if (l.owns_lock())
                {
                    next_free = next_free_tag_locked(l);
                }
            }

            if (next_free != -1)
            {
                HPX_ASSERT(next_free > 1);
                tag_provider_.release(next_free);
            }
        }

        template <typename Lock>
        int next_free_tag_locked([[maybe_unused]] Lock& lock) noexcept
        {
            HPX_ASSERT_OWNS_LOCK(lock);

            util::mpi_environment::scoped_try_lock l;
            if (l.locked)
            {
                int completed = 0;
                [[maybe_unused]] int const ret = MPI_Test(
                    &next_free_tag_request_, &completed, MPI_STATUS_IGNORE);
                HPX_ASSERT(ret == MPI_SUCCESS);

                if (completed)
                {
                    return get_next_free_tag(l);
                }
            }
            return -1;
        }

        template <typename Lock>
        int get_next_free_tag([[maybe_unused]] Lock& l) noexcept
        {
            HPX_ASSERT_OWNS_LOCK(l);

            int const next_free = next_free_tag_;

            [[maybe_unused]] int const ret = MPI_Irecv(&next_free_tag_, 1,
                MPI_INT, MPI_ANY_SOURCE, 1,
                util::mpi_environment::communicator(), &next_free_tag_request_);
            HPX_ASSERT_LOCKED(l, ret == MPI_SUCCESS);

            return next_free;
        }

        hpx::spinlock connections_mtx_;
        connection_list connections_;

        hpx::spinlock next_free_tag_mtx_;
        MPI_Request next_free_tag_request_;
        int next_free_tag_;
    };
}    // namespace hpx::parcelset::policies::mpi

#endif
