//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c) 2014-2015 Thomas Heller
//  Copyright (c) 2023 Christopher Taylor
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_OPENSHMEM)
#include <hpx/assert.hpp>
#include <hpx/modules/functional.hpp>
#include <hpx/modules/openshmem_base.hpp>
#include <hpx/modules/timing.hpp>

#include <hpx/openshmem_base/openshmem_environment.hpp>
#include <hpx/parcelport_openshmem/header.hpp>
#include <hpx/parcelport_openshmem/locality.hpp>
#include <hpx/parcelset/parcelport_connection.hpp>
#include <hpx/parcelset/parcelset_fwd.hpp>
#include <hpx/parcelset_base/detail/gatherer.hpp>
#include <hpx/parcelset_base/parcelport.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <system_error>
#include <utility>
#include <vector>

namespace hpx::parcelset::policies::openshmem {

    struct sender;
    struct sender_connection;

    //int acquire_tag(sender*) noexcept;
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
          , dst_(dst)
          , thd_id_(-1)
          , chunks_idx_(0)
          , ack_(0)
          , pp_(pp)
          , there_(parcelset::locality(locality(dst_)))
        {
            thd_id_ =
               hpx::get_worker_thread_num();    // current worker
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
            header_ = header(buffer_);
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
                const auto idx = dst_;

                std::lock_guard<hpx::mutex> l(*(*(hpx::util::openshmem_environment::segments[idx].mut)));

                // put from this localities openshmem shared memory segment
                // into the remote locality (dst_)'s shared memory segment
                //
                hpx::util::openshmem_environment::put_signal(
                    reinterpret_cast<std::uint8_t*>(header_.data()), dst_,
                    hpx::util::openshmem_environment::segments[idx].beg_addr,
                    header_.data_size_,
                    hpx::util::openshmem_environment::segments[idx].rcv
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

            std::vector<typename parcel_buffer_type::transmission_chunk_type>&
                chunks = buffer_.transmission_chunks_;

            if (!chunks.empty())
            {
                const auto idx = dst_;

                std::lock_guard<hpx::mutex> l(*(*(hpx::util::openshmem_environment::segments[idx].mut)));

                hpx::util::openshmem_environment::put_signal(
                    reinterpret_cast<std::uint8_t*>(chunks.data()), dst_,
                    hpx::util::openshmem_environment::segments[idx].beg_addr,
                    static_cast<int>(chunks.size() *
                        sizeof(parcel_buffer_type::transmission_chunk_type)),
                    hpx::util::openshmem_environment::segments[idx].rcv
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
                const auto idx = dst_;

                std::lock_guard<hpx::mutex> l(*(*(hpx::util::openshmem_environment::segments[idx].mut)));

                hpx::util::openshmem_environment::put_signal(
                    reinterpret_cast<std::uint8_t*>(buffer_.data_.data()), dst_,
                    hpx::util::openshmem_environment::segments[idx].beg_addr,
                    buffer_.data_.size(),
                    hpx::util::openshmem_environment::segments[idx].rcv
                );
            }
            state_ = sent_data;

            return send_chunks();
        }

        bool send_chunks()
        {
            HPX_ASSERT(state_ == sent_data);

            const auto idx = dst_;

            std::lock_guard<hpx::mutex> l(
                *(*(hpx::util::openshmem_environment::segments[idx].mut))
            );

            while (chunks_idx_ < buffer_.chunks_.size())
            {
                serialization::serialization_chunk& c =
                    buffer_.chunks_[chunks_idx_];
                if (c.type_ == serialization::chunk_type::chunk_type_pointer)
                {
                    if (!request_done()) {
                        return false;
                    }

                    hpx::util::openshmem_environment::put_signal(
                        reinterpret_cast<const std::uint8_t*>(c.data_.cpos_), dst_,
                        hpx::util::openshmem_environment::segments[idx].beg_addr,
                        static_cast<int>(c.size_),
                        hpx::util::openshmem_environment::segments[idx].rcv
                    );

                    hpx::util::openshmem_environment::wait_until(
                        1, hpx::util::openshmem_environment::segments[idx].xmt);
                    (*(hpx::util::openshmem_environment::segments[idx].xmt)) = 0;
                }

                ++chunks_idx_;
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
            const auto idx = dst_;

            const bool l = (*(hpx::util::openshmem_environment::segments[idx].mut))->try_lock();
            return l;

            return true;
        }

        connection_state state_;
        sender_type* sender_;
        //int tag_;
        int dst_;
        int thd_id_;

        using handler_type = hpx::move_only_function<void(error_code const&)>;
        handler_type handler_;

        using post_handler_type = hpx::move_only_function<void(
            error_code const&, parcelset::locality const&,
            std::shared_ptr<sender_connection>)>;
        post_handler_type postprocess_handler_;

        header header_;

        std::size_t chunks_idx_;
        char ack_;

        parcelset::parcelport* pp_;

        parcelset::locality there_;
    };
}    // namespace hpx::parcelset::policies::openshmem

#endif
