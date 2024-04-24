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
#include <hpx/modules/functional.hpp>
#include <hpx/modules/openshmem_base.hpp>
#include <hpx/modules/synchronization.hpp>

#include <hpx/parcelport_openshmem/sender_connection.hpp>

#include <algorithm>
#include <cstring>
#include <iterator>
#include <list>
#include <memory>
#include <mutex>
#include <utility>

namespace hpx::parcelset::policies::openshmem {

    struct sender
    {
        using connection_type = sender_connection;
        using connection_ptr = std::shared_ptr<connection_type>;
        using connection_list = std::deque<connection_ptr>;

        // different versions of clang-format disagree
        // clang-format off
        sender() noexcept
        : connections_mtx_(), connections_(), next_free_tag_mtx_()
        {
        }
        // clang-format on

        void run() noexcept
        {
            //hpx::util::openshmem_environment::scoped_lock l;
            //get_next_free_tag();
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

        void send_messages(connection_ptr connection)
        {
            // Check if sending has been completed....
            if (connection->send())
            {
                error_code ec(throwmode::lightweight);
                hpx::move_only_function<void(error_code const&,
                    parcelset::locality const&, connection_ptr)>
                    postprocess_handler;
                std::swap(
                    postprocess_handler, connection->postprocess_handler_);
                postprocess_handler(ec, connection->destination(), connection);
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
                std::unique_lock l(connections_mtx_, std::try_to_lock);
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

            return has_work;
        }

    private:

        hpx::spinlock connections_mtx_;
        connection_list connections_;

        hpx::spinlock next_free_tag_mtx_;
    };
}    // namespace hpx::parcelset::policies::openshmem

#endif
