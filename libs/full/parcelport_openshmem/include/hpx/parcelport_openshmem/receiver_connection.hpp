//  Copyright (c) 2023      Christopher Taylor
//  Copyright (c) 2014-2015 Thomas Heller
//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_OPENSHMEM)
#include <hpx/assert.hpp>
#include <hpx/modules/openshmem_base.hpp>
#include <hpx/modules/timing.hpp>

#include <hpx/parcelport_openshmem/header.hpp>
#include <hpx/parcelset/decode_parcels.hpp>
#include <hpx/parcelset/parcel_buffer.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace hpx::parcelset::policies::openshmem {

    template <typename Parcelport>
    struct receiver_connection
    {
    private:
        enum connection_state
        {
            initialized,
            rcvd_transmission_chunks,
            rcvd_data,
            rcvd_chunks
            //,sent_release_tag
        };

        using data_type = std::vector<char>;
        using buffer_type = parcel_buffer<data_type, data_type>;

    public:
        receiver_connection(int src, header h, Parcelport& pp) noexcept
          : state_(initialized)
          , src_(src)
          , header_(h)
          , request_ptr_(false)
          , num_bytes(0)
          , need_recv_data(false)
          , need_recv_tchunks(false)
          , chunks_idx_(0)
          , pp_(pp)
        {
            header_.assert_valid();

            num_bytes = header_.numbytes();

#if defined(HPX_HAVE_PARCELPORT_COUNTERS)
            parcelset::data_point& data = buffer_.data_point_;
            data.time_ = timer_.elapsed_nanoseconds();
            data.bytes_ = static_cast<std::size_t>(header_.numbytes());
#endif
            //tag_ = header_.tag();
            // decode data
            buffer_.num_chunks_ = header_.num_chunks();
            buffer_.data_.resize(static_cast<std::size_t>(header_.size()));
            char* piggy_back_data = header_.piggy_back();
            if (piggy_back_data)
            {
                need_recv_data = false;
                std::memcpy(buffer_.data_.data(), piggy_back_data,
                    buffer_.data_.size());
            }
            else
            {
                need_recv_data = true;
            }
            need_recv_tchunks = false;
        }

        bool receive()
        {
            switch (state_)
            {
            case initialized:
                return receive_transmission_chunks();

            case rcvd_transmission_chunks:
                return receive_data();

            case rcvd_data:
                return receive_chunks();

            case rcvd_chunks:
                return done();

            default:
                HPX_ASSERT(false);
            }
            return false;
        }

        bool receive_transmission_chunks()
        {
            const auto idx = hpx::util::openshmem_environment::rank();

            const std::size_t sys_pgsz =
                sysconf(_SC_PAGESIZE);

            // determine the size of the chunk buffer
            std::size_t num_zero_copy_chunks = static_cast<std::size_t>(
                static_cast<std::uint32_t>(buffer_.num_chunks_.first));
            std::size_t num_non_zero_copy_chunks = static_cast<std::size_t>(
                static_cast<std::uint32_t>(buffer_.num_chunks_.second));
            buffer_.transmission_chunks_.resize(
                num_zero_copy_chunks + num_non_zero_copy_chunks);

            if (num_zero_copy_chunks != 0)
            {
                buffer_.chunks_.resize(num_zero_copy_chunks);
                const std::size_t num_bytes =
                    buffer_.transmission_chunks_.size() * sizeof(buffer_type::transmission_chunk_type);
                const std::size_t rcv_numitrs =
                    (num_bytes + sys_pgsz - 1) / sys_pgsz;

                const std::size_t rcv_numitrs_term = rcv_numitrs - 1;

                {
                    std::size_t data_seg [2] = { sys_pgsz, num_bytes % sys_pgsz };

                    auto chunk_beg = 0;
                    for(std::size_t i = 0; i < rcv_numitrs; ++i) {
                        while(shmem_test(hpx::util::openshmem_environment::segments[idx].rcv, SHMEM_CMP_EQ, 1)) {}

    		        std::memcpy(reinterpret_cast<std::uint8_t*>(buffer_.transmission_chunks_.data())+chunk_beg,
                            hpx::util::openshmem_environment::segments[idx].beg_addr,
    			    data_seg[(i == rcv_numitrs_term)]
    		        );

                       if(i != rcv_numitrs_term) {
                            (*(hpx::util::openshmem_environment::segments[idx].rcv)) = 0;
                            chunk_beg = i * sys_pgsz;
                            hpx::util::openshmem_environment::put_signal(nullptr, src_,
                                nullptr, 0, hpx::util::openshmem_environment::segments[idx].xmt);
                        }
                    }

                    request_ptr_ = true;
                }
            }

            state_ = rcvd_transmission_chunks;

            return receive_data();
        }

        bool receive_data()
        {
            if (!request_done())
            {
                return false;
            }

            char* piggy_back = header_.piggy_back();
            if (piggy_back)
            {
                std::memcpy(
                    &buffer_.data_[0], piggy_back, buffer_.data_.size());
            }
            else
            {
                const auto idx = hpx::util::openshmem_environment::rank();

                const std::size_t sys_pgsz =
                   sysconf(_SC_PAGESIZE);

                const std::size_t num_bytes =
                    buffer_.data_.size() * sizeof(decltype(buffer_.data_);

                const std::size_t rcv_numitrs =
                    (num_bytes + sys_pgsz - 1) / sys_pgsz;

                const std::size_t rcv_numitrs_term = rcv_numitrs - 1;

                std::size_t data_seg [2] = { sys_pgsz, num_bytes % sys_pgsz };

                auto chunk_beg = 0;

                for(std::size_t i = 0; i < rcv_numitrs; ++i) {
                    while(shmem_test(hpx::util::openshmem_environment::segments[idx].rcv, SHMEM_CMP_EQ, 1)) {}

	    	    std::memcpy(reinterpret_cast<std::uint8_t*>(buffer_.transmission_chunks_.data())+chunk_beg,
                        hpx::util::openshmem_environment::segments[idx].beg_addr,
    			data_seg[(i == rcv_numitrs_term)]
		    );

                    if(i != rcv_numitrs_term) {
                       (*(hpx::util::openshmem_environment::segments[idx].rcv)) = 0;
                       chunk_beg = i * sys_pgsz;
                       hpx::util::openshmem_environment::put_signal(nullptr, src_,
                          nullptr, 0, hpx::util::openshmem_environment::segments[idx].xmt);
                    }
                }

                request_ptr_ = true;
            }

            state_ = rcvd_data;

            return receive_chunks();
        }

        bool receive_chunks()
        {
            const std::size_t sys_pgsz =
                sysconf(_SC_PAGESIZE);

            const auto idx = hpx::util::openshmem_environment::rank();

            for(auto i = 0; i < buffer_.chunks_.size(); ++i) {
                buffer_.chunks_[i].resize(buffer_.transmission_chunks_[i].second);
            }

            for(auto i = 0; i < buffer_.chunks_.size(); ++i) {
                data_type& c = buffer_.chunks_[i];

                const std::size_t num_bytes = c.size() * sizeof(decltype(c.data()));
                std::size_t data_seg [2] = { sys_pgsz, num_bytes % sys_pgsz };

                const std::size_t rcv_numitrs =
                    (num_bytes + sys_pgsz - 1) / sys_pgsz;

                const std::size_t rcv_numitrs_term = rcv_numitrs - 1;

                auto chunk_beg = 0;

                for(std::size_t i = 0; i < rcv_numitrs; ++i) {
                    while(shmem_test(hpx::util::openshmem_environment::segments[idx].rcv, SHMEM_CMP_EQ, 1)) {}
                    (*(hpx::util::openshmem_environment::segments[idx].rcv)) = 0;

                    std::memcpy(reinterpret_cast<std::uint8_t*>(c.data())+chunk_beg,
                        hpx::util::openshmem_environment::segments[idx].beg_addr,
                        data_seg[(i == rcv_numitrs_term)]
                    );

                    if(i != rcv_numitrs_term) {
                        (*(hpx::util::openshmem_environment::segments[idx].rcv)) = 0;
                        chunk_beg = i * sys_pgsz;
                        hpx::util::openshmem_environment::put_signal(nullptr, src_,
                            nullptr, 0, hpx::util::openshmem_environment::segments[idx].xmt);
                    }
                }
            }

            request_ptr_ = true;

            state_ = rcvd_chunks;

            return done();
        }

        bool done() noexcept
        {
            return request_done();
        }

        bool request_done() noexcept
        {
            util::openshmem_environment::scoped_try_lock const l;
            if(!l.locked) { return false; }

            hpx::util::openshmem_environment::quiet();

            return true;
        }

#if defined(HPX_HAVE_PARCELPORT_COUNTERS)
        hpx::chrono::high_resolution_timer timer_;
#endif
        connection_state state_;

        int src_;

        header header_;
        buffer_type buffer_;

        bool request_ptr_;
        std::size_t num_bytes;
        bool need_recv_data;
        bool need_recv_tchunks;
        std::size_t chunks_idx_;

        Parcelport& pp_;
    };
}    // namespace hpx::parcelset::policies::openshmem

#endif
