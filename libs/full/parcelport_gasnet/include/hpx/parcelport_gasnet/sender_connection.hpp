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
#include <hpx/modules/timing.hpp>

#include <hpx/parcelport_gasnet/header.hpp>
#include <hpx/parcelport_gasnet/locality.hpp>
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

namespace hpx::parcelset::policies::gasnet {
    struct sender;
    struct sender_connection;

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
        };

        using base_type =
            parcelset::parcelport_connection<sender_connection, data_type>;

    public:
        sender_connection(sender_type* s, int dst, parcelset::parcelport* pp)
          : state_(initialized)
          , sender_(s)
          , dst_rank(dst)
          , pp_(pp)
          , there_(parcelset::locality(locality(dst_rank)))
        {
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
            header_ = header(buffer_, 99);
            header_.assert_valid();

            // calculate how many long messages to send
            int long_msg_num = 0;
            // data
            if (!header_.piggy_back())
                ++long_msg_num;
            // transmission chunks
            int num_zero_copy_chunks =
                static_cast<int>(buffer_.num_chunks_.first);
            if (num_zero_copy_chunks != 0)
                long_msg_num += num_zero_copy_chunks + 1;
            
            if(num_zero_copy_chunks) {
               if(write_handles.size() != long_msg_num) {
                   write_handles.resize(long_msg_num);
               }
               
               // transmission chunk
               std::vector<
                   typename parcel_buffer_type::transmission_chunk_type>&
                   tchunks = buffer_.transmission_chunks_;
               int tchunks_length = static_cast<int>(tchunks.size() *
                   sizeof(parcel_buffer_type::transmission_chunk_type));
               write_handles[0] = gasnet_put(dst_rank, tchunks.data(), tchunks_length);

               std::size_t i = 1;
               // zero-copy chunks
               for (int j = 0; j < (int) buffer_.chunks_.size(); ++j)
               {
                   serialization::serialization_chunk& c =
                       buffer_.chunks_[j];
                   if (c.type_ ==
                       serialization::chunk_type::chunk_type_pointer)
                   {
                       auto address =
                           const_cast<void*>(c.data_.cpos_);
                       auto length = c.size_;
                       write_handles[i] = gasnet_put(dst_rank, address, length);
                       ++i;
                   }
               }
            }

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
                return done();

            default:
                HPX_ASSERT(false);
            }

            return false;
        }

        bool send_header()
        {
            if (gastnet_put(util::gasnet_environment::h_endpoint(), iovec, sync_,
                dst_rank, 0, GASNET_DEFAULT_COMP_REMOTE,
                nullptr) != GASNET_OK)
            {
                return false;
            }

            state_ = sent_header;
            return done();
        }

        bool done()
        {
            const int ret = gasnet_try_syncnb_all(write_handles.data(), write_handles.size());
            if(ret != GASNET_OK)
                return false; 

            error_code ec;
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

        connection_state state_;
        sender_type* sender_;
        int dst_rank;
        hpx::move_only_function<void(error_code const&)> handler_;
        hpx::move_only_function<void(error_code const&,
            parcelset::locality const&, std::shared_ptr<sender_connection>)>
            postprocess_handler_;

        header header_;
        parcelset::parcelport* pp_;

        parcelset::locality there_;
        std::vector<gasnet_handle_t> write_handles;
(num_zero_copy_chunks);
    };
}    // namespace hpx::parcelset::policies::gasnet

#endif
