//  Copyright (c) 2015 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_VERBS_SENDER_CONNECTION_HPP
#define HPX_PARCELSET_POLICIES_VERBS_SENDER_CONNECTION_HPP

#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/runtime/parcelset/parcelport_connection.hpp>
#include <hpx/runtime/parcelset/parcel_buffer.hpp>
//
#include <memory>
#include <cstdint>
#include <utility>
//
#include <plugins/parcelport/verbs/rdma/verbs_endpoint.hpp>
#include <plugins/parcelport/verbs/pinned_memory_vector.hpp>

namespace hpx {
namespace parcelset {
namespace policies {
namespace verbs
{
    struct sender_connection;
    struct parcelport;

    typedef header<HPX_PARCELPORT_VERBS_MESSAGE_HEADER_SIZE> header_type;
    static constexpr unsigned int header_size = header_type::header_block_size;
    //
    typedef rdma_memory_pool                                 memory_pool_type;
    typedef std::shared_ptr<memory_pool_type>                memory_pool_ptr_type;
    typedef pinned_memory_vector<char>                       snd_data_type;
    typedef parcel_buffer<snd_data_type>                     snd_buffer_type;

    struct sender_connection : std::enable_shared_from_this<sender_connection>
    {
    private:
        typedef util::function_nonser<
            void(boost::system::error_code const&, parcel const&)
            > write_handler_type;

    public:
        sender_connection(
              parcelport * pp
            , std::uint32_t dest
            , locality there
            , verbs_endpoint *client
            , memory_pool_type * chunk_pool
            , performance_counters::parcels::gatherer & parcels_sent
        )
        : buffer_(snd_buffer_type(snd_data_type(chunk_pool_), chunk_pool_))
        , parcelport_(pp)
        , dest_ip_(dest)
        , there_(there)
        , client_(client)
        , chunk_pool_(chunk_pool)
        , parcels_sent_(parcels_sent)
        {}

        snd_buffer_type get_new_buffer() {
            return snd_buffer_type(snd_data_type(chunk_pool_), chunk_pool_);
        }

        void verify_(parcelset::locality const & parcel_locality_id) const
        {
        }

        parcelset::locality const& destination() const
        {
            return there_;
        }

        bool can_send_immediate() const;

        template <typename Handler, typename ParcelPostprocess>
        void async_write(Handler && handler, ParcelPostprocess && parcel_postprocess);

        // @TODO: this buffer is never used, it is here just to shut the compiler to
        snd_buffer_type      buffer_;
        //
        parcelport          *parcelport_;
        std::uint32_t        dest_ip_;
        parcelset::locality  there_;
        verbs_endpoint      *client_;
        memory_pool_type    *chunk_pool_;
        performance_counters::parcels::gatherer & parcels_sent_;
    };
}}}}

#endif

