
//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCI)

#include <hpx/assert.hpp>
#include <hpx/parcelport_lci/header.hpp>
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
    struct buffer_wrapper
    {
        struct fake_allocator
        {
        };
        using allocator_type = fake_allocator;
        void* ptr;
        size_t length;
        buffer_wrapper() = default;
        buffer_wrapper(const buffer_wrapper& wrapper) = default;
        buffer_wrapper& operator=(const buffer_wrapper& wrapper) = default;
        buffer_wrapper(const allocator_type& alloc)
        {
            HPX_UNUSED(alloc);
            ptr = nullptr;
            length = 0;
        }
        buffer_wrapper(
            const buffer_wrapper& wrapper, const allocator_type& alloc)
        {
            HPX_UNUSED(alloc);
            ptr = wrapper.ptr;
            length = wrapper.length;
        }
        char& operator[](size_t i) const
        {
            HPX_ASSERT(i < length);
            char* p = (char*) ptr;
            return p[i];
        }
        void* data() const
        {
            return ptr;
        }
        size_t size() const
        {
            return length;
        }
    };

    struct request_wrapper_t
    {
        LCI_request_t request;
        request_wrapper_t()
        {
            request.flag = LCI_ERR_RETRY;
        }
        ~request_wrapper_t()
        {
            if (request.flag == LCI_OK)
            {
                if (request.type == LCI_IOVEC)
                {
                    for (int j = 0; j < request.data.iovec.count; ++j)
                    {
                        LCI_lbuffer_free(request.data.iovec.lbuffers[j]);
                    }
                    free(request.data.iovec.lbuffers);
                    free(request.data.iovec.piggy_back.address);
                }
                else
                {
                    HPX_ASSERT(request.type = LCI_MEDIUM);
                    LCI_mbuffer_free(request.data.mbuffer);
                }
            }
            else
            {
                HPX_ASSERT(request.flag == LCI_ERR_RETRY);
            }
        }
    };

    template <typename Parcelport>
    struct receiver
    {
        using buffer_type = parcel_buffer<buffer_wrapper, buffer_wrapper>;

        explicit receiver(Parcelport& pp) noexcept
          : pp_(pp)
        {
        }

        void run() noexcept {}

        bool background_work() noexcept
        {
            // We first try to accept a new connection
            request_wrapper_t request;
            LCI_queue_pop(util::lci_environment::get_rcq(), &request.request);

            if (request.request.flag == LCI_OK)
            {
                buffer_type buffer = decode_request(request.request);
                decode_parcels(pp_, HPX_MOVE(buffer), -1);
                return true;
            }
            return false;
        }

        buffer_type decode_request(LCI_request_t request)
        {
            buffer_type buffer_;
            header header_;
            // decode header
            if (request.type == LCI_MEDIUM)
            {
                // only header
                header_ = header((char*) request.data.mbuffer.address);
                header_.assert_valid();

                HPX_ASSERT(
                    header_.piggy_back_data() && !header_.piggy_back_tchunk());
                HPX_ASSERT(header_.num_zero_copy_chunks() == 0);
            }
            else
            {
                // iovec
                HPX_ASSERT(request.type == LCI_IOVEC);
                header_ = header((char*) request.data.iovec.piggy_back.address);
                header_.assert_valid();
            }
#if defined(HPX_HAVE_PARCELPORT_COUNTERS)
            hpx::chrono::high_resolution_timer timer_;
            parcelset::data_point& data = buffer_.data_point_;
            data.time_ = timer_.elapsed_nanoseconds();
            data.bytes_ = static_cast<std::size_t>(header_.numbytes());
#endif
            int i = 0;
            // decode data
            char* piggy_back_data = header_.piggy_back_data();
            if (piggy_back_data)
            {
                buffer_.data_.length = header_.numbytes_nonzero_copy();
                buffer_.data_.ptr = piggy_back_data;
            }
            else
            {
                HPX_ASSERT(request.type == LCI_IOVEC);
                HPX_ASSERT((size_t) header_.numbytes_nonzero_copy() ==
                    request.data.iovec.lbuffers[i].length);
                buffer_.data_.length = header_.numbytes_nonzero_copy();
                buffer_.data_.ptr = request.data.iovec.lbuffers[i].address;
                ++i;
            }
            // decode transmission chunk
            int num_zero_copy_chunks = header_.num_zero_copy_chunks();
            int num_non_zero_copy_chunks = header_.num_non_zero_copy_chunks();
            buffer_.num_chunks_.first = num_zero_copy_chunks;
            buffer_.num_chunks_.second = num_non_zero_copy_chunks;
            if (num_zero_copy_chunks != 0)
            {
                HPX_ASSERT(request.type == LCI_IOVEC);
                auto& tchunks = buffer_.transmission_chunks_;
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
                    HPX_ASSERT((size_t) tchunks_length ==
                        request.data.iovec.lbuffers[i].length);
                    std::memcpy((void*) tchunks.data(),
                        request.data.iovec.lbuffers[i].address, tchunks_length);
                    ++i;
                }
                // zero-copy chunks
                buffer_.chunks_.resize(num_zero_copy_chunks);
                for (int j = 0; j < num_zero_copy_chunks; ++j)
                {
                    std::size_t chunk_size =
                        buffer_.transmission_chunks_[j].second;
                    HPX_ASSERT(
                        request.data.iovec.lbuffers[i].length == chunk_size);
                    buffer_wrapper& c = buffer_.chunks_[j];
                    c.length = chunk_size;
                    c.ptr = request.data.iovec.lbuffers[i].address;
                    ++i;
                }
            }
            HPX_ASSERT(
                request.type == LCI_MEDIUM || i == request.data.iovec.count);

#if defined(HPX_HAVE_PARCELPORT_COUNTERS)
            data.time_ = timer_.elapsed_nanoseconds() - data.time_;
#endif
            return buffer_;
        }

        Parcelport& pp_;
    };

}    // namespace hpx::parcelset::policies::lci

#endif
