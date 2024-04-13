//  Copyright (c) 2023-2024 Jiakun Yan
//  Copyright (c) 2007-2023 Hartmut Kaiser
//  Copyright (c) 2014-2015 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCI)

#include <hpx/assert.hpp>
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
    struct receiver_putva : public receiver_base
    {
        explicit receiver_putva(parcelport* pp) noexcept
          : receiver_base(pp)
        {
        }

        bool background_work() noexcept
        {
            bool did_some_work = false;

            auto poll_comp_start = util::lci_environment::pcounter_now();
            auto completion_manager_p =
                pp_->get_tls_device().completion_manager_p;
            request_wrapper_t request;
            request.request = completion_manager_p->recv_new->poll();
            util::lci_environment::pcounter_add(
                util::lci_environment::poll_comp,
                util::lci_environment::pcounter_since(poll_comp_start));

            if (request.request.flag == LCI_OK)
            {
                auto useful_bg_start = util::lci_environment::pcounter_now();
                HPX_ASSERT(request.request.flag == LCI_OK);
                process_request(request.request);
                util::lci_environment::pcounter_add(
                    util::lci_environment::useful_bg_work,
                    util::lci_environment::pcounter_since(useful_bg_start));
                did_some_work = true;
            }
            return did_some_work;
        }

    private:
        void process_request(LCI_request_t request)
        {
            if (request.type == LCI_MEDIUM)
            {
                size_t consumed = 0;
                while (consumed < request.data.mbuffer.length)
                {
                    buffer_type buffer;
                    consumed += decode_eager(
                        (char*) request.data.mbuffer.address + consumed,
                        buffer);
                    handle_received_parcels(
                        decode_parcels(*pp_, HPX_MOVE(buffer)));
                }
                HPX_ASSERT(consumed == request.data.mbuffer.length);
            }
            else
            {
                // iovec
                HPX_ASSERT(request.type == LCI_IOVEC);
                buffer_type buffer;
                decode_iovec(request.data.iovec, buffer);
                handle_received_parcels(decode_parcels(*pp_, HPX_MOVE(buffer)));
            }
        }

        size_t decode_eager(void* address, buffer_type& buffer)
        {
#if defined(HPX_HAVE_PARCELPORT_COUNTERS)
            hpx::chrono::high_resolution_timer timer_;
            parcelset::data_point& data = buffer.data_point_;
#endif
            util::lci_environment::pcounter_add(
                util::lci_environment::recv_conn_start, 1);
            // decode header
            header header_ = header((char*) address);
            header_.assert_valid();
            HPX_ASSERT(
                header_.piggy_back_data() && !header_.piggy_back_tchunk());
            HPX_ASSERT(header_.num_zero_copy_chunks() == 0);    // decode data
            // decode data
            buffer.data_.length = header_.numbytes_nonzero_copy();
            buffer.data_.ptr = header_.piggy_back_data();
            // decode transmission chunk
            int num_zero_copy_chunks = header_.num_zero_copy_chunks();
            int num_non_zero_copy_chunks = header_.num_non_zero_copy_chunks();
            buffer.num_chunks_.first = num_zero_copy_chunks;
            buffer.num_chunks_.second = num_non_zero_copy_chunks;
            util::lci_environment::pcounter_add(
                util::lci_environment::recv_conn_end, 1);
#if defined(HPX_HAVE_PARCELPORT_COUNTERS)
            data.bytes_ = static_cast<std::size_t>(header_.numbytes());
            data.time_ = timer_.elapsed_nanoseconds() - data.time_;
#endif
            return header_.size();
        }

        void decode_iovec(LCI_iovec_t iovec, buffer_type& buffer)
        {
#if defined(HPX_HAVE_PARCELPORT_COUNTERS)
            hpx::chrono::high_resolution_timer timer_;
            parcelset::data_point& data = buffer.data_point_;
            data.time_ = timer_.elapsed_nanoseconds();
#endif
            util::lci_environment::pcounter_add(
                util::lci_environment::recv_conn_start, 1);
            // decode header
            header header_ = header((char*) iovec.piggy_back.address);
            header_.assert_valid();
            int i = 0;
            // decode data
            char* piggy_back_data = header_.piggy_back_data();
            if (piggy_back_data)
            {
                buffer.data_.length = header_.numbytes_nonzero_copy();
                buffer.data_.ptr = piggy_back_data;
            }
            else
            {
                HPX_ASSERT((size_t) header_.numbytes_nonzero_copy() ==
                    iovec.lbuffers[i].length);
                buffer.data_.length = header_.numbytes_nonzero_copy();
                buffer.data_.ptr = iovec.lbuffers[i].address;
                ++i;
            }
            if (header_.num_zero_copy_chunks() != 0)
            {
                // decode transmission chunk
                int num_zero_copy_chunks = header_.num_zero_copy_chunks();
                int num_non_zero_copy_chunks =
                    header_.num_non_zero_copy_chunks();
                buffer.num_chunks_.first = num_zero_copy_chunks;
                buffer.num_chunks_.second = num_non_zero_copy_chunks;
                auto& tchunks = buffer.transmission_chunks_;
                tchunks.resize(num_zero_copy_chunks + num_non_zero_copy_chunks);
                int tchunks_length = static_cast<int>(tchunks.size() *
                    sizeof(buffer_type::transmission_chunk_type));
                char* piggy_back_tchunk = header_.piggy_back_tchunk();
                if (piggy_back_tchunk)
                {
                    std::memcpy((void*) tchunks.data(), piggy_back_tchunk,
                        tchunks_length);
                }
                else
                {
                    HPX_ASSERT(
                        (size_t) tchunks_length == iovec.lbuffers[i].length);
                    std::memcpy((void*) tchunks.data(),
                        iovec.lbuffers[i].address, tchunks_length);
                    ++i;
                }
                // zero-copy chunks
                buffer.chunks_.resize(num_zero_copy_chunks);
                for (int j = 0; j < num_zero_copy_chunks; ++j)
                {
                    std::size_t chunk_size =
                        buffer.transmission_chunks_[j].second;
                    HPX_ASSERT(iovec.lbuffers[i].length == chunk_size);
                    buffer.chunks_[j] = serialization::create_pointer_chunk(
                        iovec.lbuffers[i].address, chunk_size);
                    ++i;
                }
            }
            HPX_ASSERT(i == iovec.count);
            util::lci_environment::pcounter_add(
                util::lci_environment::recv_conn_end, 1);
#if defined(HPX_HAVE_PARCELPORT_COUNTERS)
            data.bytes_ = static_cast<std::size_t>(header_.numbytes());
            data.time_ = timer_.elapsed_nanoseconds();
#endif
        }
    };

}    // namespace hpx::parcelset::policies::lci

#endif
