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
        bool tryMerge(
            const std::shared_ptr<sender_connection_base>& other_base);

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
        return_t unified_followup_send(void* address, int length);
        return_t send_transmission_chunks();
        return_t send_data();
        return_t send_chunks();
        // the state of this connection
        std::atomic<connection_state> state;
        size_t send_chunks_idx;
        // related information about this connection
        hpx::chrono::high_resolution_timer timer_;
        header header_;
        LCI_mbuffer_t header_buffer;
        std::vector<char> header_buffer_vector;
        bool need_send_data;
        bool need_send_tchunks;
        LCI_tag_t tag;
        LCI_tag_t original_tag;
        std::shared_ptr<sender_connection_sendrecv>* sharedPtr_p;
        // temporary data
        LCI_comp_t completion;
        LCI_segment_t segment_to_use, segment_used;
        // for profiling
        LCT_time_t conn_start_time;

        static std::atomic<int> next_tag;
    };
}    // namespace hpx::parcelset::policies::lci

#endif
