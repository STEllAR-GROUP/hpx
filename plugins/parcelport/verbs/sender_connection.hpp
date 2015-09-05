//  Copyright (c) 2015 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_VERBS_SENDER_CONNECTION_HPP
#define HPX_PARCELSET_POLICIES_VERBS_SENDER_CONNECTION_HPP

#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/runtime/parcelset/parcelport_connection.hpp>
#include <hpx/runtime/parcelset/parcel_buffer.hpp>
#include <hpx/util/memory_chunk_pool_allocator.hpp>
//#include <hpx/plugins/parcelport/verbs/locality.hpp>

#include <boost/shared_ptr.hpp>
#include "RdmaClient.h"

namespace hpx { namespace parcelset {
namespace policies { namespace verbs
{
    struct sender_connection;
    struct parcelport;

    typedef lcos::local::spinlock                           mutex_type;
    typedef char                                            memory_type;
    typedef RdmaMemoryPool                                  memory_pool_type;
    typedef std::shared_ptr<memory_pool_type>               memory_pool_ptr_type;
    typedef util::detail::memory_chunk_pool_allocator
            <memory_type, memory_pool_type, mutex_type>     allocator_type;

    typedef std::vector<memory_type, allocator_type>        snd_data_type;
    typedef parcel_buffer<snd_data_type>                    snd_buffer_type;

    struct sender_connection
      : parcelset::parcelport_connection<
            sender_connection
          , snd_data_type
        >
    {
    private:
        typedef parcelport parcelport_type;

        typedef util::function_nonser<
            void(boost::system::error_code const&, parcel const&)
        > write_handler_type;


        typedef
            parcelset::parcelport_connection<sender_connection, snd_data_type>
            base_type;

    public:
        sender_connection(
            parcelport_type * pp
          , boost::uint32_t dest
          , locality there
          , RdmaClient *client
          , memory_pool_type & chunk_pool
          , performance_counters::parcels::gatherer & parcels_sent
        )
          : base_type(allocator_type(chunk_pool))
          , parcelport_(pp)
          , dest_ip_(dest)
          , there_(there)
          , client_(client)
          , chunk_pool_(chunk_pool)
          , parcels_sent_(parcels_sent)
        {
            // the send buffer is created with our allocator and will get memory from our pool
            // - disable deallocation so that we can manage the block lifetime better
            // @TODO, integrate the pointer wrapper and allocators better into parcel_buffer
            allocator_type alloc(chunk_pool_);
            alloc.disable_deallocate = true;
            snd_buffer_type buffer(alloc);
            buffer_ = std::move(buffer);
        }

        void verify(parcelset::locality const & parcel_locality_id) const
        {
        }

        parcelset::locality const& destination() const
        {
            return there_;
        }

        template <typename Handler, typename ParcelPostprocess>
        void async_write(Handler && handler, ParcelPostprocess && parcel_postprocess);

        util::unique_function_nonser< void(error_code const&) > handler_;

        util::unique_function_nonser< void(
                error_code const&
              , parcelset::locality const&
              , boost::shared_ptr<sender_connection>
            )
        > postprocess_handler_;

        parcelport_type * parcelport_;
        boost::uint32_t dest_ip_;
        parcelset::locality there_;
        RdmaClient *client_;
        memory_pool_type & chunk_pool_;
        performance_counters::parcels::gatherer & parcels_sent_;
    };
}}}}

#endif

