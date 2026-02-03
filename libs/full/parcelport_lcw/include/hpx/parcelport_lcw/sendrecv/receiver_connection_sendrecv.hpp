//  Copyright (c) 2025 Jiakun Yan
//  Copyright (c) 2014-2015 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCW)

#include <hpx/parcelport_lcw/parcelport_lcw.hpp>
#include <hpx/parcelport_lcw/receiver_base.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <system_error>
#include <utility>
#include <vector>

namespace hpx::parcelset::policies::lcw {
    struct receiver_connection_sendrecv
      : public std::enable_shared_from_this<receiver_connection_sendrecv>
    {
        struct return_t
        {
            bool isDone;
            ::lcw::comp_t completion;
        };

    public:
        receiver_connection_sendrecv(int dst, parcelset::parcelport* pp);
        ~receiver_connection_sendrecv() {}
        void load(char* header_buffer);
        return_t receive();
        void done();

    private:
        enum class connection_state
        {
            initialized,
            rcvd_transmission_chunks,
            rcvd_data,
            rcvd_chunks,
            locked
        };
        ::lcw::comp_t unified_recv(void* address, size_t length);
        return_t receive_transmission_chunks();
        return_t receive_data();
        return_t receive_chunks();
        void receive_chunks_zc_preprocess();
        return_t receive_chunks_zc();
        return_t receive_chunks_nzc();
        // the state of this connection
        std::atomic<connection_state> state;
        size_t recv_chunks_idx;
        size_t recv_zero_copy_chunks_idx;
        // related information about this connection
        hpx::chrono::high_resolution_timer timer_;
        int dst_rank;
        bool need_recv_data;
        bool need_recv_tchunks;
        ::lcw::tag_t tag;
        ::lcw::tag_t original_tag;
        receiver_base::buffer_type buffer;
        std::vector<parcelset::parcel> parcels_;
        std::vector<std::vector<char>> chunk_buffers_;
        parcelport* pp_;
        parcelport::device_t* device_p;
        std::shared_ptr<receiver_connection_sendrecv>* sharedPtr_p;
        // for profiling
        LCT_time_t conn_start_time;
    };
}    // namespace hpx::parcelset::policies::lcw

#endif
