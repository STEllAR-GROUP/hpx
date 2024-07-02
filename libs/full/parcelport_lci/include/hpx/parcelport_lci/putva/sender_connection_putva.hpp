//  Copyright (c) 2023-2024 Jiakun Yan
//  Copyright (c) 2014-2015 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCI)

#include <hpx/parcelport_lci/sender_connection_base.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <system_error>
#include <utility>
#include <vector>

namespace hpx::parcelset::policies::lci {
    struct sender_connection_putva : public sender_connection_base
    {
    public:
        sender_connection_putva(int dst, parcelset::parcelport* pp)
          : sender_connection_base(dst, pp)
        {
        }
        ~sender_connection_putva() {}
        void load(handler_type&& handler,
            postprocess_handler_type&& parcel_postprocess);
        return_t send_nb();
        void done();
        bool tryMerge(
            const std::shared_ptr<sender_connection_base>& other_base);

    private:
        enum class connection_state
        {
            initialized,
            sent,
            locked,
        };
        bool can_be_eager_message(size_t max_header_size);
        bool isEager();
        void cleanup();
        return_t send_msg();

        std::atomic<connection_state> state;
        bool is_eager;
        LCI_mbuffer_t mbuffer;
        LCI_iovec_t iovec;
        std::shared_ptr<sender_connection_putva>*
            sharedPtr_p;    // for LCI_putva
        // for profiling
        LCT_time_t conn_start_time;
    };
}    // namespace hpx::parcelset::policies::lci

#endif
