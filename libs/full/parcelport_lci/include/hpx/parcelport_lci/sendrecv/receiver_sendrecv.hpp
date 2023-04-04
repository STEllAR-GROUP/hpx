
//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCI)

#include <hpx/parcelport_lci/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/parcelport_lci/parcelport_lci.hpp>
#include <hpx/parcelport_lci/receiver_base.hpp>
#include <hpx/parcelset/decode_parcels.hpp>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <deque>
#include <iterator>
#include <list>
#include <memory>
#include <mutex>
#include <set>
#include <utility>
#include <vector>

namespace hpx::parcelset::policies::lci {
    struct receiver_connection_sendrecv;
    struct receiver_sendrecv : public receiver_base
    {
        using connection_type = receiver_connection_sendrecv;
        using connection_ptr = std::shared_ptr<connection_type>;

        explicit receiver_sendrecv(parcelport* pp) noexcept
          : receiver_base(pp)
        {
            if (config_t::protocol == config_t::protocol_t::sendrecv)
            {
                for (int i = 0; i < config_t::prepost_recv_num; ++i)
                {
                    LCI_comp_t completion =
                        pp_->recv_new_completion_manager->alloc_completion();
                    LCI_recvmn(pp_->endpoint_new_eager, LCI_RANK_ANY, 0,
                        completion, nullptr);
                    pp_->recv_new_completion_manager->enqueue_completion(
                        completion);
                }
            }
        }

        ~receiver_sendrecv() {}

        connection_ptr create_connection(int dest, parcelset::parcelport* pp);

        bool background_work() noexcept;

    private:
        bool accept_new() noexcept;
        bool followup() noexcept;
    };

}    // namespace hpx::parcelset::policies::lci

#endif
