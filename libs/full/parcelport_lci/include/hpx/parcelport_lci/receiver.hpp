
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
#include <hpx/parcelport_lci/receiver_connection.hpp>

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
    template <typename Parcelport>
    struct receiver
    {
        using mutex_type = hpx::spinlock;
        using header_list = std::list<std::pair<int, header>>;
        using handles_header_type = std::set<std::pair<int, int>>;
        using connection_type = receiver_connection<Parcelport>;
        using connection_ptr = std::shared_ptr<connection_type>;
        using connection_list = std::deque<connection_ptr>;
        using data_type = std::vector<char>;
        using buffer_type = parcel_buffer<data_type, data_type>;

        explicit receiver(Parcelport& pp) noexcept
          : pp_(pp)
        {
        }

        void run() noexcept {}

        bool background_work() noexcept
        {
            // We first try to accept a new connection
            LCI_request_t request = accept();

            if (request.flag == LCI_OK)
            {
                buffer_type buffer = decode_request(request);
                decode_parcels(pp_, HPX_MOVE(buffer), -1);
                return true;
            }
            return false;
            //            // If we don't have a new connection, try to handle one of the
            //            // already accepted ones.
            //            if (!connection)
            //            {
            //                std::unique_lock<mutex_type> l(
            //                    connections_mtx_, std::try_to_lock);
            //                if (l.owns_lock() && !connections_.empty())
            //                {
            //                    connection = HPX_MOVE(connections_.front());
            //                    connections_.pop_front();
            //                }
            //            }
            //
            //            if (connection)
            //            {
            //                receive_messages(HPX_MOVE(connection));
            //                return true;
            //            }
            //
            //            return false;
        }

        buffer_type decode_request(LCI_request_t request)
        {
            buffer_type buffer_;
            header header_;
            // decode header
            if (request.type == LCI_MEDIUM)
            {
                // only header
                header_ = *(header*) (request.data.mbuffer.address);
                header_.assert_valid();

                HPX_ASSERT(header_.piggy_back());
                HPX_ASSERT(header_.num_chunks().first == 0);
            }
            else
            {
                // iovec
                HPX_ASSERT(request.type == LCI_IOVEC);
                header_ = *(header*) (request.data.iovec.piggy_back.address);
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
            buffer_.data_.resize(header_.size());
            char* piggy_back = header_.piggy_back();
            if (piggy_back)
            {
                std::memcpy(&buffer_.data_[0], piggy_back, header_.size());
            }
            else
            {
                HPX_ASSERT(request.type == LCI_IOVEC);
                HPX_ASSERT((size_t) header_.size() ==
                    request.data.iovec.lbuffers[i].length);
                std::memcpy(&buffer_.data_[0],
                    request.data.iovec.lbuffers[i].address, header_.size());
                ++i;
            }
            // decode transmission chunk
            buffer_.num_chunks_ = header_.num_chunks();
            int num_zero_copy_chunks =
                static_cast<int>(buffer_.num_chunks_.first);
            int num_non_zero_copy_chunks =
                static_cast<int>(buffer_.num_chunks_.second);
            auto& tchunks = buffer_.transmission_chunks_;
            tchunks.resize(num_zero_copy_chunks + num_non_zero_copy_chunks);
            if (num_zero_copy_chunks != 0)
            {
                HPX_ASSERT(request.type == LCI_IOVEC);
                buffer_.chunks_.resize(num_zero_copy_chunks);
                int tchunks_length = static_cast<int>(tchunks.size() *
                    sizeof(buffer_type::transmission_chunk_type));
                HPX_ASSERT((size_t) tchunks_length ==
                    request.data.iovec.lbuffers[i].length);
                std::memcpy((void*) tchunks.data(),
                    request.data.iovec.lbuffers[i].address, tchunks_length);
                ++i;
                // zero-copy chunks
                for (int j = 0; j < num_zero_copy_chunks; ++j)
                {
                    std::size_t chunk_size =
                        buffer_.transmission_chunks_[j].second;
                    HPX_ASSERT(
                        request.data.iovec.lbuffers[i].length == chunk_size);
                    data_type& c = buffer_.chunks_[j];
                    c.resize(chunk_size);
                    std::memcpy((void*) c.data(),
                        request.data.iovec.lbuffers[i].address, chunk_size);
                    ++i;
                }
            }
            HPX_ASSERT(
                request.type == LCI_MEDIUM || i == request.data.iovec.count);

#if defined(HPX_HAVE_PARCELPORT_COUNTERS)
            data.time_ = timer_.elapsed_nanoseconds() - data.time_;
#endif
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
                LCI_mbuffer_free(request.data.mbuffer);
            }

            return buffer_;
        }

        LCI_request_t accept() noexcept
        {
            //            connection_ptr res;
            LCI_request_t request;
            request.flag = LCI_ERR_RETRY;
            LCI_queue_pop(util::lci_environment::h_queue(), &request);
            //                header h = *(header*) (request.data.mbuffer.address);
            //                h.assert_valid();
            //
            //                res.reset(new connection_type(request.rank, h, pp_));
            //                LCI_mbuffer_free(request.data.mbuffer);
            return request;
        }

        Parcelport& pp_;

        mutex_type headers_mtx_;
        header rcv_header_;

        mutex_type handles_header_mtx_;
        handles_header_type handles_header_;

        mutex_type connections_mtx_;
        connection_list connections_;
    };

}    // namespace hpx::parcelset::policies::lci

#endif
