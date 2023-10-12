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
#include <hpx/modules/functional.hpp>
#include <hpx/modules/gasnet_base.hpp>
#include <hpx/modules/synchronization.hpp>

#include <hpx/parcelport_gasnet/sender_connection.hpp>
#include <hpx/parcelport_gasnet/tag_provider.hpp>

#include <algorithm>
#include <cstring>
#include <iterator>
#include <list>
#include <memory>
#include <mutex>
#include <utility>

namespace hpx::parcelset::policies::gasnet {

    struct sender
    {
        using connection_type = sender_connection;
        using connection_ptr = std::shared_ptr<connection_type>;
        using connection_list = std::deque<connection_ptr>;

        // different versions of clang-format disagree
        // clang-format off
        sender() noexcept
          : next_free_tag_(-1)
        {
        }
        // clang-format on

        void run() noexcept
        {
            hpx::util::gasnet_environment::scoped_lock l;
            get_next_free_tag();
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
            next_free_tag();

            return has_work;
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
                    next_free = next_free_tag_locked();
                }
            }

            if (next_free != -1)
            {
                HPX_ASSERT(next_free > 1);
                tag_provider_.release(next_free);
            }
        }

        int next_free_tag_locked() noexcept
        {
            hpx::util::gasnet_environment::scoped_try_lock l;
            if (l.locked)
            {
                return get_next_free_tag();
            }
            return -1;
        }

        int get_next_free_tag() noexcept
        {
            int next_free = next_free_tag_;

            hpx::util::gasnet_environment::scoped_lock l;
            std::memcpy(&next_free,
                hpx::util::gasnet_environment::segments
                    [hpx::util::gasnet_environment::rank()]
                        .addr,
                sizeof(int));

            return next_free;
        }

        hpx::spinlock connections_mtx_;
        connection_list connections_;

        hpx::spinlock next_free_tag_mtx_;
        int next_free_tag_;
    };

}    // namespace hpx::parcelset::policies::gasnet

#endif
