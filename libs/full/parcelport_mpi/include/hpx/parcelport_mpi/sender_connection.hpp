//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c) 2014-2015 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_MPI)
#include <hpx/assert.hpp>
#include <hpx/modules/functional.hpp>
#include <hpx/modules/mpi_base.hpp>
#include <hpx/parcelport_mpi/header.hpp>
#include <hpx/parcelport_mpi/locality.hpp>
#include <hpx/parcelset/parcelport_connection.hpp>
#include <hpx/parcelset/parcelset_fwd.hpp>
#include <hpx/parcelset_base/detail/gatherer.hpp>
#include <hpx/parcelset_base/parcelport.hpp>
#if defined(HPX_HAVE_PARCELPORT_COUNTERS)
#include <hpx/modules/timing.hpp>
#endif

#include <cstddef>
#include <cstdint>
#include <memory>
#include <system_error>
#include <utility>
#include <vector>

namespace hpx::parcelset::policies::mpi {

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
          , request_(MPI_REQUEST_NULL)
          , request_ptr_(nullptr)
          , chunks_idx_(0)
          , ack_(0)
          , pp_(pp)
          , there_(parcelset::locality(locality(dst_)))
        {
        }

        constexpr parcelset::locality const& destination() const noexcept
        {
            return there_;
        }

        static constexpr void verify_(
            parcelset::locality const& /* parcel_locality_id */) noexcept
        {
        }

        using handler_type = hpx::move_only_function<void(error_code const&)>;
        using post_handler_type = hpx::move_only_function<void(
            error_code const&, parcelset::locality const&,
            std::shared_ptr<sender_connection>)>;
        void async_write(
            handler_type&& handler, post_handler_type&& parcel_postprocess)
        {
            HPX_ASSERT(!handler_);
            HPX_ASSERT(!buffer_.data_.empty());

#if defined(HPX_HAVE_PARCELPORT_COUNTERS)
            buffer_.data_point_.time_ = static_cast<std::int64_t>(
                hpx::chrono::high_resolution_clock::now());
#endif
            request_ptr_ = nullptr;
            chunks_idx_ = 0;
            tag_ = acquire_tag(sender_);
            header_.init(buffer_, tag_);
            header_.assert_valid();

            state_ = initialized;

            handler_ = HPX_MOVE(handler);

            if (!send())
            {
                postprocess_handler_ = HPX_MOVE(parcel_postprocess);
                add_connection(sender_, shared_from_this());
            }
            else
            {
                HPX_ASSERT(!handler_);
                error_code ec;
                if (parcel_postprocess)
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
                util::mpi_environment::scoped_lock l;
                HPX_ASSERT(state_ == initialized);
                HPX_ASSERT(request_ptr_ == nullptr);

                [[maybe_unused]] int const ret = MPI_Isend(header_.data(),
                    header::data_size_, MPI_BYTE, dst_, 0,
                    util::mpi_environment::communicator(), &request_);
                HPX_ASSERT_LOCKED(l, ret == MPI_SUCCESS);

                request_ptr_ = &request_;
            }

            state_ = sent_header;
            return send_transmission_chunks();
        }

        bool send_transmission_chunks()
        {
            HPX_ASSERT(state_ == sent_header);
            HPX_ASSERT(request_ptr_ != nullptr);
            if (!request_done())
            {
                return false;
            }

            HPX_ASSERT(request_ptr_ == nullptr);

            auto const& chunks = buffer_.transmission_chunks_;
            if (!chunks.empty())
            {
                util::mpi_environment::scoped_lock l;

                [[maybe_unused]] int const ret = MPI_Isend(chunks.data(),
                    static_cast<int>(chunks.size() *
                        sizeof(parcel_buffer_type::transmission_chunk_type)),
                    MPI_BYTE, dst_, tag_, util::mpi_environment::communicator(),
                    &request_);
                HPX_ASSERT_LOCKED(l, ret == MPI_SUCCESS);

                request_ptr_ = &request_;
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
                util::mpi_environment::scoped_lock l;

                [[maybe_unused]] int const ret = MPI_Isend(buffer_.data_.data(),
                    static_cast<int>(buffer_.data_.size()), MPI_BYTE, dst_,
                    tag_, util::mpi_environment::communicator(), &request_);
                HPX_ASSERT_LOCKED(l, ret == MPI_SUCCESS);

                request_ptr_ = &request_;
            }
            state_ = sent_data;

            return send_chunks();
        }

        bool send_chunks()
        {
            HPX_ASSERT(state_ == sent_data);

            while (chunks_idx_ < buffer_.chunks_.size())
            {
                auto const& c = buffer_.chunks_[chunks_idx_];
                if (c.type_ == serialization::chunk_type::chunk_type_pointer)
                {
                    if (!request_done())
                    {
                        return false;
                    }

                    util::mpi_environment::scoped_lock l;

                    [[maybe_unused]] int const ret =
                        MPI_Isend(const_cast<void*>(c.data_.cpos_),
                            static_cast<int>(c.size_), MPI_BYTE, dst_, tag_,
                            util::mpi_environment::communicator(), &request_);
                    HPX_ASSERT_LOCKED(l, ret == MPI_SUCCESS);

                    request_ptr_ = &request_;
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

            error_code const ec(throwmode::lightweight);
            handler_(ec);
            handler_.reset();
#if defined(HPX_HAVE_PARCELPORT_COUNTERS)
            buffer_.data_point_.time_ =
                static_cast<std::int64_t>(
                    hpx::chrono::high_resolution_clock::now()) -
                buffer_.data_point_.time_;
            pp_->add_sent_data(buffer_.data_point_);
#endif
            buffer_.clear();

            state_ = initialized;

            return true;
        }

        bool request_done()
        {
            if (request_ptr_ == nullptr)
            {
                return true;
            }

            util::mpi_environment::scoped_try_lock const l;
            if (!l.locked)
            {
                return false;
            }

            int completed = 0;
            [[maybe_unused]] int const ret =
                MPI_Test(request_ptr_, &completed, MPI_STATUS_IGNORE);
            HPX_ASSERT(ret == MPI_SUCCESS);
            if (completed)
            {
                request_ptr_ = nullptr;
                return true;
            }
            return false;
        }

        connection_state state_;
        sender_type* sender_;
        int tag_;
        int dst_;

        handler_type handler_;
        post_handler_type postprocess_handler_;

        header header_;

        MPI_Request request_;
        MPI_Request* request_ptr_;
        std::size_t chunks_idx_;
        char ack_;

        parcelset::parcelport* pp_;

        parcelset::locality there_;
    };
}    // namespace hpx::parcelset::policies::mpi

#endif
