//  Copyright (c) 2023-2024 Jiakun Yan
//  Copyright (c) 2014-2023 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCI)

#include <deque>
#include <memory>
#include <vector>

namespace hpx::parcelset::policies::lci {
    struct completion_manager_base;
    struct sender_connection_base;
    namespace backlog_queue {
        using message_type = sender_connection_base;
        using message_ptr = std::shared_ptr<message_type>;
        struct backlog_queue_t
        {
            // pending messages per destination
            std::vector<std::deque<message_ptr>> messages;
        };

        void push(message_ptr message);
        bool empty(int dst_rank);
        bool background_work(completion_manager_base* completion_manager,
            size_t num_thread) noexcept;
        void free();
    }    // namespace backlog_queue
}    // namespace hpx::parcelset::policies::lci

#endif
