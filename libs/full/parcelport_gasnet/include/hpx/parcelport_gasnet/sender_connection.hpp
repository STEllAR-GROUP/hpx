//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c) 2014-2015 Thomas Heller
//  Copyright (c) 2023 Christopher Taylor
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_GASNET)
#include <hpx/assert.hpp>
#include <hpx/modules/functional.hpp>
#include <hpx/modules/gasnet_base.hpp>
#include <hpx/modules/timing.hpp>

#include <hpx/parcelport_gasnet/header.hpp>
#include <hpx/parcelport_gasnet/locality.hpp>
#include <hpx/parcelset/parcelport_connection.hpp>
#include <hpx/parcelset/parcelset_fwd.hpp>
#include <hpx/parcelset_base/detail/gatherer.hpp>
#include <hpx/parcelset_base/parcelport.hpp>
#include <hpx/gasnet_base/gasnet_environment.hpp>

#include <cstddef>
#include <memory>
#include <system_error>
#include <utility>
#include <vector>

namespace hpx::parcelset::policies::gasnet {

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
          , dst_(dst)
          , chunks_idx_(0)
          , ack_(0)
          , pp_(pp)
          , there_(parcelset::locality(locality(dst_)))
        {
           gasnet_buffer.resize(hpx::util::gasnet_environment::size());
           for(std::size_t i = 0; i < gasnet_buffer.size(); ++i) {
               gasnet_buffer[i] = static_cast<std::uint8_t*>(hpx::util::gasnet_environment::segments[i].addr);
           }
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

#if defined(HPX_HAVE_PARCELPORT_COUNTERS)
            buffer_.data_point_.time_ =
                hpx::chrono::high_resolution_clock::now();
#endif
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
                util::gasnet_environment::scoped_lock l;
                HPX_ASSERT(state_ == initialized);

                // compute + send the number of GASNET_PAGEs to send and the remainder number of bytes to a GASNET_PAGE
                //
                const std::size_t chunks[] = { header_.data_size_ / GASNET_PAGESIZE, header_.data_size_ % GASNET_PAGESIZE };
                std::copy(chunks, sizeof(chunks), gasnet_buffer[hpx::util::gasnet_environment::rank()]);

                hpx::util::gasnet_environment::put<std::uint8_t>(
                    gasnet_buffer[hpx::util::gasnet_environment::rank()],
                    dst_,
                    gasnet_buffer[dst_],
                    sizeof(chunks)
                );
            }

            state_ = sent_header;
            return send_transmission_chunks();
        }

        bool send_transmission_chunks()
        {
            HPX_ASSERT(state_ == sent_header);
            if (!request_done())
            {
                return false;
            }

            HPX_ASSERT(request_ptr_ == nullptr);

            std::vector<typename parcel_buffer_type::transmission_chunk_type>&
                chunks = buffer_.transmission_chunks_;
            if (!chunks.empty())
            {
                util::gasnet_environment::scoped_lock l;
                std::copy_n(chunks.data(), chunks.size(), gasnet_buffer[hpx::util::gasnet_environment::rank()]);
                gasnet_put_bulk(dst_, gasnet_buffer[dst_], gasnet_buffer[hpx::util::gasnet_environment::rank()],
                    static_cast<int>(chunks.size() * sizeof(parcel_buffer_type::transmission_chunk_type))
                );
            }

            state_ = sent_transmission_chunks;
            return send_data();
        }

        bool send_data()
        {
            HPX_ASSERT(state_ == sent_transmission_chunks);
            if (!request_done())
            {
                return false;
            }

            if (!header_.piggy_back())
            {
                util::gasnet_environment::scoped_lock l;
                std::copy_n(buffer_.data_.data(), buffer_.data_.size(), gasnet_buffer[hpx::util::gasnet_environment::rank()]);

                hpx::util::gasnet_environment::put<std::uint8_t>(
                    gasnet_buffer[hpx::util::gasnet_environment::rank()],
                    dst_,
                    gasnet_buffer[dst_],
                    sizeof(chunks)
                );
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
                    {
                        return false;
                    }

                    util::gasnet_environment::scoped_lock l;

                    std::copy_n(const_cast<std::uint8_t*>(c.data_.c_pos_), static_cast<int>(c.size_), gasnet_buffer[hpx::util::gasnet_environment::rank()]);

                    hpx::util::gasnet_environment::put<std::uint8_t>(
                        gasnet_buffer[hpx::util::gasnet_environment::rank()],
                        dst_,
                        gasnet_buffer[dst_],
                        static_cast<int>(c.size_)
                    );
                }

                chunks_idx_++;
            }

            state_ = sent_chunks;

            return done();
        }

        bool done()
        {
            if (!request_done())
            {
                return false;
            }

            error_code ec(throwmode::lightweight);
            handler_(ec);
            handler_.reset();
#if defined(HPX_HAVE_PARCELPORT_COUNTERS)
            buffer_.data_point_.time_ =
                hpx::chrono::high_resolution_clock::now() -
                buffer_.data_point_.time_;
            pp_->add_sent_data(buffer_.data_point_);
#endif
            buffer_.clear();

            state_ = initialized;

            return true;
        }

        bool request_done()
        {
            util::gasnet_environment::scoped_try_lock l;
            if (!l.locked)
            {
                return false;
            }

            return true;
        }

        connection_state state_;
        sender_type* sender_;
        int tag_;
        int dst_;
        hpx::move_only_function<void(error_code const&)> handler_;
        hpx::move_only_function<void(error_code const&,
            parcelset::locality const&, std::shared_ptr<sender_connection>)>
            postprocess_handler_;

        header header_;
        std::vector<std::uint8_t *> gasnet_buffer;

        std::size_t chunks_idx_;
        char ack_;

        parcelset::parcelport* pp_;

        parcelset::locality there_;
    };
}    // namespace hpx::parcelset::policies::gasnet

#endif
