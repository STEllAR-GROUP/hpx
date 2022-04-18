//  Copyright (c) 2014-2015 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCI)
#include <hpx/assert.hpp>
#include <hpx/modules/functional.hpp>
#include <hpx/modules/timing.hpp>

#include <hpx/parcelport_lci/header.hpp>
#include <hpx/parcelport_lci/locality.hpp>
#include <hpx/parcelset/parcelport_connection.hpp>
#include <hpx/parcelset/parcelset_fwd.hpp>
#include <hpx/parcelset_base/detail/gatherer.hpp>
#include <hpx/parcelset_base/parcelport.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <system_error>
#include <utility>
#include <vector>

namespace hpx::parcelset::policies::lci {
    struct sender;
    struct sender_connection;

    int acquire_tag(sender*) noexcept;
    void add_connection(sender*, std::shared_ptr<sender_connection> const&);

    struct sender_connection
      : parcelset::parcelport_connection<sender_connection, std::vector<char>>
    {
    private:
        using sender_type = sender;

        using write_handler_type =
            hpx::function<void(std::error_code const&, parcel const&)>;

        using data_type = std::vector<char>;

        enum connection_state
        {
            initialized,
            sent_header,
            sent_transmission_chunks,
            sent_data,
            sent_chunks
        };

        using base_type =
            parcelset::parcelport_connection<sender_connection, data_type>;

    public:
        sender_connection(sender_type* s, int dst, parcelset::parcelport* pp)
          : state_(initialized)
          , sender_(s)
          , tag_(-1)
          , dst_rank(dst)
          , request_ptr_(nullptr)
          , chunks_idx_(0)
          , ack_(0)
          , pp_(pp)
          , there_(parcelset::locality(locality(dst_rank)))
        {
            LCI_sync_create(LCI_UR_DEVICE, 1, &sync_);
        }

        parcelset::locality const& destination() const noexcept
        {
            return there_;
        }

        constexpr void verify_(
            parcelset::locality const& /* parcel_locality_id */) const noexcept
        {
        }

        template <typename Handler, typename ParcelPostprocess>
        void async_write(
            Handler&& handler, ParcelPostprocess&& parcel_postprocess)
        {
            HPX_ASSERT(!handler_);
            HPX_ASSERT(!postprocess_handler_);
            HPX_ASSERT(!buffer_.data_.empty());

            buffer_.data_point_.time_ =
                hpx::chrono::high_resolution_clock::now();
            request_ptr_ = nullptr;
            chunks_idx_ = 0;
            tag_ = acquire_tag(sender_);
            header_ = header(buffer_, tag_);
            header_.assert_valid();

            state_ = initialized;

            handler_ = HPX_FORWARD(Handler, handler);

            if (!send())
            {
                postprocess_handler_ =
                    HPX_FORWARD(ParcelPostprocess, parcel_postprocess);
                add_connection(sender_, shared_from_this());
            }
            else
            {
                HPX_ASSERT(!handler_);
                error_code ec;
                parcel_postprocess(ec, there_, shared_from_this());
            }
        }

        bool send()
        {
            switch (state_)
            {
            case initialized:
                return send_header();

            case sent_header:
                return send_transmission_chunks();

            case sent_transmission_chunks:
                return send_data();

            case sent_data:
                return send_chunks();

            case sent_chunks:
                return done();

            default:
                HPX_ASSERT(false);
            }

            return false;
        }

        bool send_header()
        {
            {
                HPX_ASSERT(state_ == initialized);
                HPX_ASSERT(request_ptr_ == nullptr);
                HPX_ASSERT(LCI_MEDIUM_SIZE >= header_.data_size_);
                LCI_mbuffer_t mbuffer;
                mbuffer.length = header_.data_size_;
                mbuffer.address = header_.data();
                if (LCI_putma(util::lci_environment::h_endpoint(), mbuffer,
                        dst_rank, 0, LCI_DEFAULT_COMP_REMOTE) != LCI_OK)
                {
                    return false;
                }
            }

            state_ = sent_header;
            return send_transmission_chunks();
        }

        bool unified_send(void* buffer, int length, int rank, LCI_tag_t tag)
        {
            LCI_error_t ret;
            if (length <= LCI_MEDIUM_SIZE)
            {
                LCI_mbuffer_t mbuffer;
                mbuffer.address = buffer;
                mbuffer.length = length;
                ret = LCI_sendm(
                    util::lci_environment::lci_endpoint(), mbuffer, rank, tag);
            }
            else
            {
                LCI_lbuffer_t lbuffer;
                lbuffer.address = buffer;
                lbuffer.length = length;
                lbuffer.segment = LCI_SEGMENT_ALL;
                ret = LCI_sendl(util::lci_environment::lci_endpoint(), lbuffer,
                    rank, tag, sync_, nullptr);
                if (ret == LCI_OK)
                {
                    request_ptr_ = &sync_;
                }
            }
            return ret == LCI_OK;
        }

        bool send_transmission_chunks()
        {
            HPX_ASSERT(state_ == sent_header);
            HPX_ASSERT(request_ptr_ == nullptr);
            std::size_t num_zero_copy_chunks = static_cast<std::size_t>(
                static_cast<std::uint32_t>(buffer_.num_chunks_.first));

            std::vector<typename parcel_buffer_type::transmission_chunk_type>&
                tchunks = buffer_.transmission_chunks_;
            // this makes me very confused
            // the original condition here is !tchunks.empty()
            // the receive counterpart has the condition num_zero_copy_chunks != 0
            if (num_zero_copy_chunks != 0)
            {
                int tchunks_length = static_cast<int>(tchunks.size() *
                    sizeof(parcel_buffer_type::transmission_chunk_type));
                bool ret = unified_send(
                    tchunks.data(), tchunks_length, dst_rank, tag_);
                if (!ret)
                    return false;
            }
            state_ = sent_transmission_chunks;
            return send_data();
        }

        bool send_data()
        {
            HPX_ASSERT(state_ == sent_transmission_chunks);
            if (!request_done())
                return false;

            if (!header_.piggy_back())
            {
                bool ret = unified_send(buffer_.data_.data(),
                    static_cast<int>(buffer_.data_.size()), dst_rank, tag_);
                if (!ret)
                    return false;
            }
            state_ = sent_data;
            return send_chunks();
        }

        bool send_chunks()
        {
            HPX_ASSERT(state_ == sent_data);

            while (chunks_idx_ < buffer_.chunks_.size())
            {
                serialization::serialization_chunk& c =
                    buffer_.chunks_[chunks_idx_];
                if (c.type_ == serialization::chunk_type::chunk_type_pointer)
                {
                    if (!request_done())
                        return false;
                    else
                    {
                        bool ret =
                            unified_send(const_cast<void*>(c.data_.cpos_),
                                static_cast<int>(c.size_), dst_rank, tag_);
                        if (!ret)
                            return false;
                    }
                }
                chunks_idx_++;
            }
            state_ = sent_chunks;

            return done();
        }

        bool request_done()
        {
            if (request_ptr_ == nullptr)
                return true;
            HPX_ASSERT(request_ptr_ == &sync_);

            LCI_error_t ret = LCI_sync_test(sync_, nullptr);
            if (ret == LCI_OK)
            {
                request_ptr_ = nullptr;
                return true;
            }
            else
            {
                return false;
            }
        }

        bool done()
        {
            if (!request_done())
                return false;

            error_code ec;
            handler_(ec);
            handler_.reset();
            buffer_.data_point_.time_ =
                hpx::chrono::high_resolution_clock::now() -
                buffer_.data_point_.time_;
            pp_->add_sent_data(buffer_.data_point_);
            buffer_.clear();

            state_ = initialized;

            return true;
        }

        connection_state state_;
        sender_type* sender_;
        int tag_;
        int dst_rank;
        hpx::move_only_function<void(error_code const&)> handler_;
        hpx::move_only_function<void(error_code const&,
            parcelset::locality const&, std::shared_ptr<sender_connection>)>
            postprocess_handler_;

        header header_;

        void* request_ptr_;
        LCI_comp_t sync_;
        std::size_t chunks_idx_;
        char ack_;

        parcelset::parcelport* pp_;

        parcelset::locality there_;
    };
}    // namespace hpx::parcelset::policies::lci

#endif
