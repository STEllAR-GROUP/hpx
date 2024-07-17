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

    void add_connection(sender*, std::shared_ptr<sender_connection> const&);

    struct sender_connection
      : parcelset::parcelport_connection<sender_connection, std::vector<char>>
    {
    private:
	using buffer_type = std::vector<char>;
	using buffer_value_type = std::vector<char>::value_type;
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
	  , src_(0)
          , chunks_idx_(0)
          , ack_(0)
          , pp_(pp)
          , there_(parcelset::locality(locality(dst_)))
        {
std::cout << "sender_connection_instantiated" << std::endl;

	    src_ = hpx::util::openshmem_environment::rank();
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
std::cout << "send" << std::endl;
            switch (state_)
            {
            case initialized:
std::cout << "sent_init" << std::endl;
                return send_header();

            case sent_header:
std::cout << "sent_header" << std::endl;
                return send_transmission_chunks();

            case sent_transmission_chunks:
std::cout << "sent_transmission_chunks" << std::endl;
                return send_data();

            case sent_data:
std::cout << "sent_data" << std::endl;
                return send_chunks();

            case sent_chunks:
std::cout << "sent_chunks" << std::endl;
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

                const std::size_t sys_pgsz =
                    sysconf(_SC_PAGESIZE);
                const std::size_t page_count =
                    hpx::util::openshmem_environment::size();
                const std::size_t beg_rcv_signal =
                    (sys_pgsz*page_count);

                // header_.data_size_*sizeof(decltype(header_.data)),
                const std::size_t data_amt = sizeof(header);
                std::size_t data_seg [2] = { sys_pgsz, data_amt % sys_pgsz };

                const std::size_t header_numitrs =
                    (sizeof(header) + sys_pgsz - 1) / sys_pgsz;

                const std::size_t header_numitrs_term = header_numitrs - 1;

std::cout << "sendHeader0\t" << dst_ << ' ' << std::endl << std::flush;
                //util::openshmem_environment::scoped_lock l;
                //HPX_ASSERT_OWNS_LOCK(header_lock);
                HPX_ASSERT(state_ == initialized);
std::cout << "sendHeader1" << std::endl << std::flush;

                // put from this localities openshmem shared memory segment
                // into the remote locality (dst_)'s shared memory segment
                //
                auto amt = 0;
                for(std::size_t itr = 0; itr < header_numitrs; ++itr) {
                    hpx::util::openshmem_environment::put_signal(
                        reinterpret_cast<std::uint8_t*>(header_.data()) + amt, dst_,
                        hpx::util::openshmem_environment::segments[src_].beg_addr,
                        data_seg[(itr == header_numitrs_term)],
                        hpx::util::openshmem_environment::segments[src_].rcv
                    );

                    if(itr != header_numitrs_term) { 
		        amt = data_seg[(itr == header_numitrs_term)] * itr;
                        while(hpx::util::openshmem_environment::test(hpx::util::openshmem_environment::segments[dst_].xmt, 1)) {}
                        (*(hpx::util::openshmem_environment::segments[dst_].xmt)) = 0;
                    }
                }
            }

            state_ = sent_header;
            return send_transmission_chunks();
        }

        bool send_transmission_chunks()
        {
std::cout << "send_transmission_chunks" << std::endl;
            HPX_ASSERT(state_ == sent_header);
            if (!request_done())
            {
                return false;
            }

            std::vector<typename parcel_buffer_type::transmission_chunk_type>&
                chunks = buffer_.transmission_chunks_;

            if (!chunks.empty())
            {
                //util::openshmem_environment::scoped_lock l;

                const std::size_t sys_pgsz =
                    sysconf(_SC_PAGESIZE);
                const std::size_t page_count =
                    hpx::util::openshmem_environment::size();
                const std::size_t beg_rcv_signal =
                    (sys_pgsz*page_count);

                const std::size_t data_amt =
                    static_cast<int>(chunks.size() *
                        sizeof(parcel_buffer_type::transmission_chunk_type));

                std::size_t data_seg [2] = { sys_pgsz, data_amt % sys_pgsz };

                const std::size_t header_numitrs =
                    ((data_amt + sys_pgsz - 1) / sys_pgsz) + data_seg[1] ? 0 : - 1;

                const std::size_t header_numitrs_term = header_numitrs - 1;

		hpx::util::openshmem_environment::scoped_lock l;

                // put from this localities openshmem shared memory segment
                // into the remote locality (dst_)'s shared memory segment
                //
                auto amt = 0;
                for(std::size_t itr = 0; itr < header_numitrs; ++itr) {
                    hpx::util::openshmem_environment::put_signal(
                        reinterpret_cast<std::uint8_t*>(chunks.data()) + amt, dst_,
                        hpx::util::openshmem_environment::segments[src_].beg_addr,
                        data_seg[(itr == header_numitrs_term)],
                        hpx::util::openshmem_environment::segments[src_].rcv
                    );

                    if(itr != header_numitrs_term) { 
		        amt = data_seg[(itr == header_numitrs_term)] * itr;
                        while(hpx::util::openshmem_environment::test(hpx::util::openshmem_environment::segments[dst_].xmt, 1)) {}
                        (*(hpx::util::openshmem_environment::segments[dst_].xmt)) = 0;
                    }
                }
            }

            state_ = sent_transmission_chunks;
            return send_data();
        }

        bool send_data()
        {
std::cout << "send_data" << std::endl;
            HPX_ASSERT(state_ == sent_transmission_chunks);
            if (!request_done())
            {
                return false;
            }

            if (!header_.piggy_back())
            {   
                //util::openshmem_environment::scoped_lock l;
		std::cout << "is null" << dst_ << ' ' << (hpx::util::openshmem_environment::segments[src_].beg_addr == nullptr) << ' ' << (hpx::util::openshmem_environment::segments[src_].rcv == nullptr) << std::endl;

// issue here is page size < data size

                const std::size_t sys_pgsz =
                    sysconf(_SC_PAGESIZE);
                const std::size_t page_count =
                    hpx::util::openshmem_environment::size();
                const std::size_t beg_rcv_signal =
                    (sys_pgsz*page_count);

                const std::size_t data_amt =
                    buffer_.data_.size() * sizeof(buffer_value_type);

                std::size_t data_seg [2] = { sys_pgsz, data_amt % sys_pgsz };

                const std::size_t header_numitrs =
                    ((data_amt + sys_pgsz - 1) / sys_pgsz) + data_seg[1] ? 0 : - 1;

                const std::size_t header_numitrs_term = header_numitrs - 1;

		hpx::util::openshmem_environment::scoped_lock l;

                // put from this localities openshmem shared memory segment
                // into the remote locality (dst_)'s shared memory segment
                //
                auto amt = 0;
                for(std::size_t itr = 0; itr < header_numitrs; ++itr) {
                    hpx::util::openshmem_environment::put_signal(
                        reinterpret_cast<std::uint8_t*>(buffer_.data_.data() + amt), dst_,
                        hpx::util::openshmem_environment::segments[src_].beg_addr,
                        data_seg[(itr == header_numitrs_term)],
                        hpx::util::openshmem_environment::segments[src_].rcv
                    );

                    if(itr != header_numitrs_term) {
		        amt = data_seg[(itr == header_numitrs_term)] * itr;
                        while(hpx::util::openshmem_environment::test(hpx::util::openshmem_environment::segments[dst_].xmt, 1)) {}
                        (*(hpx::util::openshmem_environment::segments[dst_].xmt)) = 0;
                    }
                }
            }
            state_ = sent_data;

            return send_chunks();
        }

        bool send_chunks()
        {
std::cout << "send_chunks" << std::endl;
            HPX_ASSERT(state_ == sent_data);

            while (chunks_idx_ < buffer_.chunks_.size())
            {
                serialization::serialization_chunk& c =
                    buffer_.chunks_[chunks_idx_];
                if (c.type_ == serialization::chunk_type::chunk_type_pointer)
                {
                    if (!request_done()) {
                        return false;
                    }

                    {
                        //util::openshmem_environment::scoped_lock l;

                        const std::size_t sys_pgsz =
                            sysconf(_SC_PAGESIZE);
                        const std::size_t page_count =
                            hpx::util::openshmem_environment::size();
                        const std::size_t beg_rcv_signal =
                            (sys_pgsz*page_count);

                        const std::size_t data_amt =
                            static_cast<int>(c.size_);

                        std::size_t data_seg [2] = { sys_pgsz, data_amt % sys_pgsz };

                        const std::size_t header_numitrs =
                            ((data_amt + sys_pgsz - 1) / sys_pgsz) + data_seg[1] ? 0 : - 1;

                        const std::size_t header_numitrs_term = header_numitrs - 1;

                        // put from this localities openshmem shared memory segment
                        // into the remote locality (dst_)'s shared memory segment
                        //
                        auto amt = 0;
                        for(std::size_t itr = 0; itr < header_numitrs; ++itr) {
                            hpx::util::openshmem_environment::put_signal(
                                    reinterpret_cast<const std::uint8_t*>(c.data_.cpos_ + amt), dst_,
                                hpx::util::openshmem_environment::segments[src_].beg_addr,
                                data_seg[(itr == header_numitrs_term)],
                                hpx::util::openshmem_environment::segments[src_].rcv
                            );

                            if(itr != header_numitrs_term) {
		                amt = data_seg[(itr == header_numitrs_term)] * itr;
                                while(hpx::util::openshmem_environment::test(hpx::util::openshmem_environment::segments[dst_].xmt, 1)) {}
                                (*(hpx::util::openshmem_environment::segments[dst_].xmt)) = 0;
                            }
                        }
                    }
                }

                ++chunks_idx_;
            }

            state_ = sent_chunks;

            return done();
        }

        bool done()
        {
std::cout << "done" << std::endl;
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
std::cout << "request_done1" << std::endl;
            //util::openshmem_environment::scoped_try_lock const l;
            //if(!l.locked) { return false; }

            hpx::util::openshmem_environment::fence();
std::cout << "request_done2" << std::endl;

            return true;
        }

        connection_state state_;
        sender_type* sender_;
        int dst_;
        int src_;

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
