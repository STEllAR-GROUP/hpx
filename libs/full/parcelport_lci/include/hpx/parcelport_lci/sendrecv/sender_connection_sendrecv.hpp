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
    struct sender_connection_sendrecv : public sender_connection_base
    {
    public:
        sender_connection_sendrecv(int dst, parcelset::parcelport* pp)
          : sender_connection_base(dst, pp)
        {
        }
        ~sender_connection_sendrecv() {}
        void load(handler_type&& handler,
            postprocess_handler_type&& parcel_postprocess);
        return_t send_nb();
        void done();

    private:
        enum class connection_state
        {
            initialized,
            sent_header,
            sent_transmission_chunks,
            sent_data,
            sent_chunks,
            locked,
        };
        return_t send_header();
        return_t unified_followup_send(void* address, size_t length);
        return_t send_transmission_chunks();
        return_t send_data();
        return_t send_chunks();
        // the state of this connection
        std::atomic<connection_state> state;
        size_t send_chunks_idx;
        // related information about this connection
        hpx::chrono::high_resolution_timer timer_;
        header header_;
        void* header_buffer;
        size_t header_buffer_size;
        bool need_send_data;
        bool need_send_tchunks;
        ::lci::tag_t tag;
        ::lci::tag_t original_tag;
        std::shared_ptr<sender_connection_sendrecv>* sharedPtr_p;
        // temporary data
        ::lci::comp_t completion;
        // for profiling
        LCT_time_t conn_start_time;

        static std::atomic<::lci::tag_t> next_tag;
    };
}    // namespace hpx::parcelset::policies::lci

#endif
