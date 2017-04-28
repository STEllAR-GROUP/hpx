//  Copyright (c) 2015-2017 John Biddiscombe
//  Copyright (c) 2017      Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <plugins/parcelport/libfabric/rma_receiver.hpp>
#include <plugins/parcelport/libfabric/parcelport_libfabric.hpp>
//
#include <hpx/runtime/parcelset/parcel_buffer.hpp>
#include <hpx/runtime/parcelset/decode_parcels.hpp>
//
#include <hpx/util/detail/yield_k.hpp>
//
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace hpx {
namespace parcelset {
namespace policies {
namespace libfabric
{
// --------------------------------------------------------------------
    rma_receiver::rma_receiver(
        parcelport* pp,
        fid_ep* endpoint,
        rma_memory_pool<region_provider>* memory_pool,
        completion_handler&& handler)
      : pp_(pp),
        endpoint_(endpoint),
        header_region_(nullptr),
        message_region_(nullptr),
        memory_pool_(memory_pool),
        handler_(std::move(handler)),
        rma_count_(0)
    {}

    // --------------------------------------------------------------------
    rma_receiver::~rma_receiver()
    {
        LOG_DEBUG_MSG("Receiving of message complete " << hexpointer(this));
    }

    // --------------------------------------------------------------------
    void rma_receiver::read_message(region_type* rrregion,
        fi_addr_t const& src_addr)
    {
        start_time_ = util::high_resolution_clock::now();
        HPX_ASSERT(rma_count_ == 0);
        HPX_ASSERT(header_ == nullptr);
        HPX_ASSERT(header_region_ == nullptr);
        HPX_ASSERT(message_region_ == nullptr);
        HPX_ASSERT(rma_regions_.size() == 0);

        // where this message came from
        src_addr_ = src_addr;

        // the region posted as a receive contains the received header
        header_region_ = rrregion;
        header_        = reinterpret_cast<header_type*>(header_region_->get_address());

        HPX_ASSERT(header_);
        HPX_ASSERT(header_region_->get_address());

        LOG_DEBUG_MSG("receiver " << hexpointer(this) << "Header : " << *header_);

        LOG_TRACE_MSG(
            CRC32_MEM(header_, header_->header_length(), "Header region (recv)"));

        // how mand RMA operations are needed
        rma_count_ = header_->num_zero_copy_chunks();

        LOG_DEBUG_MSG("receiver " << hexpointer(this)
            << "is expecting " << decnumber(rma_count_) << "read completions");

        // If we have no zero copy chunks and piggy backed data, we can
        // process the message immediately, otherwise, dispatch to receiver
        // If we have neither piggy back, nor zero copy chunks, rma_count is 0
        if (rma_count_ == 0)
        {
            handle_message_no_rma();
            ++msg_plain_;
        }
        else {
            handle_message_with_zerocopy_rma();
            ++msg_rma_;
        }
    }

    // --------------------------------------------------------------------
    void rma_receiver::handle_message_no_rma()
    {
        HPX_ASSERT(header_);
        LOG_DEBUG_MSG("receiver " << hexpointer(this)
            << "handle piggy backed send without zero copy regions");

        char *piggy_back = header_->message_data();
        HPX_ASSERT(piggy_back);

        LOG_TRACE_MSG(
            CRC32_MEM(piggy_back, header_->message_size(),
                "(Message region recv piggybacked - no rdma)"));

        typedef pinned_memory_vector<char, header_size, region_type, memory_pool_type>
            rcv_data_type;
        typedef parcel_buffer<rcv_data_type, std::vector<char>> rcv_buffer_type;

        // when parcel decoding from the wrapped pointer buffer has completed,
        // the lambda function will be called
        rcv_data_type wrapped_pointer(
            piggy_back, header_->message_size(), [](){}, nullptr, nullptr);

        rcv_buffer_type buffer(std::move(wrapped_pointer), nullptr);
        buffer.num_chunks_ = std::make_pair(
            header_->num_zero_copy_chunks(), header_->num_index_chunks());
        buffer.data_size_  = header_->message_size();
        performance_counters::parcels::data_point& data = buffer.data_point_;
        data.bytes_ = static_cast<std::size_t>(header_->message_size());
        data.time_ = util::high_resolution_clock::now() - start_time_;
        LOG_DEBUG_MSG("receiver " << hexpointer(this)
            << "calling parcel decode for complete NORMAL parcel");
        std::size_t num_thread = hpx::get_worker_thread_num();
        decode_message_with_chunks(*pp_, std::move(buffer), 1, chunks_, num_thread);
        LOG_DEBUG_MSG("receiver " << hexpointer(this)
            << "parcel decode called for complete NORMAL (small) parcel");

        cleanup_receive();
    }

    // --------------------------------------------------------------------
    void rma_receiver::handle_message_with_zerocopy_rma()
    {
        chunks_.resize(header_->num_chunks());
        char *chunk_data = header_->chunk_data();
        HPX_ASSERT(chunk_data);

        size_t chunkbytes =
            chunks_.size() * sizeof(chunk_struct);

        std::memcpy(chunks_.data(), chunk_data, chunkbytes);
        LOG_DEBUG_MSG("receiver " << hexpointer(this)
            << "Copied chunk data from header : size "
            << decnumber(chunkbytes));

        LOG_EXCLUSIVE(
        for (const chunk_struct &c : chunks_)
        {
            LOG_DEBUG_MSG("receiver " << hexpointer(this)
                << "recv : chunk : size " << hexnumber(c.size_)
                << " type " << decnumber((uint64_t)c.type_)
                << " rkey " << hexpointer(c.rkey_)
                << " cpos " << hexpointer(c.data_.cpos_)
                << " index " << decnumber(c.data_.index_));
        });

        rma_regions_.reserve(header_->num_zero_copy_chunks());

        // for each zerocopy chunk, schedule a read operation
        std::size_t index = 0;
        for (chunk_struct &c : chunks_)
        {
            if (c.type_ == serialization::chunk_type_pointer)
            {
                region_type *get_region =
                    memory_pool_->allocate_region(c.size_);
                LOG_TRACE_MSG(
                    CRC32_MEM(get_region->get_address(), c.size_,
                        "(RDMA GET region (new))"));

                LOG_DEBUG_MSG("receiver " << hexpointer(this)
                    << "RDMA Get addr " << hexpointer(c.data_.cpos_)
                    << "rkey " << hexpointer(c.rkey_)
                    << "size " << hexnumber(c.size_)
                    << "tag " << hexuint64(header_->tag())
                    << "local addr " << hexpointer(get_region->get_address())
                    << "length " << hexlength(c.size_));

                rma_regions_.push_back(get_region);
                get_region->set_message_length(c.size_);

                // overwrite the serialization chunk data to account for the
                // local pointers instead of remote ones
                const void *remoteAddr = c.data_.cpos_;
                c.data_.cpos_ = get_region->get_address();

                // post the rdma read/get
                LOG_DEBUG_MSG("receiver " << hexpointer(this)
                    << "RDMA Get fi_read :"
                    << "chunk " << decnumber(index)
                    << "client " << hexpointer(endpoint_)
                    << "fi_addr " << hexpointer(src_addr_)
                    << "local addr " << hexpointer(get_region->get_address())
                    << "local desc " << hexpointer(get_region->get_desc())
                    << "size " << hexlength(c.size_)
                    << "rkey " << hexpointer(c.rkey_)
                    << "remote cpos " << hexpointer(remoteAddr));

                // count reads
                ++rma_reads_;

                ssize_t ret = 0;
                for (std::size_t k = 0; true; ++k)
                {
                    LOG_EXCLUSIVE(
                        // write a pattern and dump out data for debugging purposes
                        uint32_t *buffer =
                            reinterpret_cast<uint32_t*>(get_region->get_address());
                        std::fill(buffer, buffer + get_region->get_size()/4, 0x01010101);
                        LOG_TRACE_MSG(
                            CRC32_MEM(get_region->get_address(), c.size_,
                                "(RDMA GET region (pre-fi_read))"));
                    );

                    ret = fi_read(endpoint_,
                        get_region->get_address(), c.size_, get_region->get_desc(),
                        src_addr_,
                        (uint64_t)(remoteAddr), c.rkey_, this);
                    if (ret == -FI_EAGAIN)
                    {
                        LOG_ERROR_MSG("receiver " << hexpointer(this)
                            << "reposting fi_read...\n");
                        hpx::util::detail::yield_k(k, "libfabric::receiver::async_read");
                        continue;
                    }
                    if (ret) throw fabric_error(ret, "fi_read");
                    break;
                }
            }
            ++index;
        }
    }

    // --------------------------------------------------------------------
    void rma_receiver::handle_rma_read_completion()
    {
        FUNC_START_DEBUG_MSG;
        HPX_ASSERT(rma_count_ > 0);
        // If we haven't read all chunks, we can return and wait
        // for the other incoming read completions
        if (--rma_count_ > 0)
        {
            LOG_DEBUG_MSG("receiver " << hexpointer(this)
                << "Not yet read all RMA regions " << hexpointer(this));
            FUNC_START_DEBUG_MSG;
            return;
        }

        HPX_ASSERT(rma_count_ == 0);
        LOG_DEBUG_MSG("receiver " << hexpointer(this)
            << "all RMA regions now read ");

        // If the main message was not piggy backed, then the final chunk
        // is our main message block, remove it from the rma list
        if (!header_->message_piggy_back())
        {
            message_region_ = rma_regions_.back();
            //
            rma_regions_.resize(rma_regions_.size()-1);
            chunks_.resize(chunks_.size()-1);
        }

        std::size_t message_length = header_->message_size();
        char *message = nullptr;
        if (message_region_)
        {
            message = static_cast<char *>(message_region_->get_address());
            HPX_ASSERT(message);
            LOG_DEBUG_MSG("receiver " << hexpointer(this)
                << "No piggy_back RDMA message "
                << "region " << hexpointer(message_region_)
                << "address " << hexpointer(message_region_->get_address())
                << "length " << hexuint32(message_length));
            LOG_TRACE_MSG(
                CRC32_MEM(message, message_length, "Message region (recv rdma)"));

            // do this after dumping out data as otherwise we lose some debug info
            HPX_ASSERT(message_region_->get_message_length() == header_->message_size());
        }
        else
        {
            HPX_ASSERT(header_->message_data());
            message = header_->message_data();
            LOG_TRACE_MSG(CRC32_MEM(message, message_length,
                "Message region (recv piggyback with rdma)"));
        }

        for (auto &r : rma_regions_)
        {
            LOG_TRACE_MSG(CRC32_MEM(r->get_address(), r->get_message_length(),
                "rdma region (recv) "));
        }

        // wrap the message and chunks into a pinned vector so that they
        // can be passed into the parcel decode functions and when released have
        // the pinned buffers returned to the memory pool
        typedef pinned_memory_vector<char, header_size, region_type, memory_pool_type>
            rcv_data_type;
        typedef parcel_buffer<rcv_data_type, std::vector<char>> rcv_buffer_type;

        rcv_data_type wrapped_pointer(message, message_length,
            [this, message, message_length]()
            {
                // deleted cleanup until all problems resolved, now after parcel decode
                if (message_region_) {
                    LOG_TRACE_MSG(CRC32_MEM(message, message_length,
                        "Message region (receiver delete)"));
                }
            }, nullptr, nullptr);

        rcv_buffer_type buffer(std::move(wrapped_pointer), nullptr);

        LOG_EXCLUSIVE(
            for (chunk_struct &c : chunks_) {
                LOG_DEBUG_MSG("get : chunk : size " << hexnumber(c.size_)
                    << " type " << decnumber((uint64_t)c.type_)
                    << " rkey " << hexpointer(c.rkey_)
                    << " cpos " << hexpointer(c.data_.cpos_)
                    << " index " << decnumber(c.data_.index_));
        });

        buffer.num_chunks_ = std::make_pair(
            header_->num_original_zero_copy_chunks(), header_->num_index_chunks());
        buffer.data_size_  = header_->message_size();
        performance_counters::parcels::data_point& data = buffer.data_point_;
        data.bytes_ = static_cast<std::size_t>(header_->message_size());
        data.time_ = util::high_resolution_clock::now() - start_time_;

        LOG_DEBUG_MSG("receiver " << hexpointer(this)
            << "calling parcel decode for ZEROCOPY complete parcel");
        std::size_t num_thread = hpx::get_worker_thread_num();
        decode_message_with_chunks(*pp_, std::move(buffer), 1, chunks_, num_thread);
        LOG_DEBUG_MSG("receiver " << hexpointer(this)
            << "parcel decode called for ZEROCOPY complete parcel");

        LOG_DEBUG_MSG("receiver " << hexpointer(this) << "Sending ack");
        send_rdma_complete_ack();

        cleanup_receive();
        FUNC_END_DEBUG_MSG;
    }

    // --------------------------------------------------------------------
    void rma_receiver::send_rdma_complete_ack()
    {
        LOG_DEBUG_MSG("receiver " << hexpointer(this)
            << "RDMA Get tag " << hexuint64(header_->tag())
            << " has completed : posting 8 byte ack to origin");

        ++sent_ack_;

        int ret = 0;
        for (std::size_t k = 0; true; ++k)
        {
            // when we received the incoming message, the tag was already set
            // with the sender context so that we can signal it directly
            // that we have completed RMA and the sender my now cleanup.
            std::uint64_t tag = header_->tag();
            ret = fi_inject(endpoint_, &tag, sizeof(std::uint64_t), src_addr_);
            if (ret == -FI_EAGAIN)
            {
                LOG_ERROR_MSG("receiver " << hexpointer(this)
                    << "reposting fi_inject...\n");
                hpx::util::detail::yield_k(k,
                    "libfabric::receiver::send_rdma_complete_ack");
                continue;
            }
            if (ret) throw fabric_error(ret, "fi_inject ack notification error");
            break;
        }
    }

    // --------------------------------------------------------------------
    void rma_receiver::cleanup_receive()
    {
        LOG_DEBUG_MSG("cleanup for receiver rma " << hexpointer(this));
        //
        HPX_ASSERT(rma_count_ == 0);
        //
        ++recv_deletes_;
        //
        memory_pool_->deallocate(header_region_);
        header_region_ = nullptr;
        header_        = nullptr;
        src_addr_      = 0 ;
        //
        if (message_region_) {
            memory_pool_->deallocate(message_region_);
            message_region_ = nullptr;
        }
        //
        for (auto region: rma_regions_) {
            memory_pool_->deallocate(region);
        }
        rma_regions_.clear();
        chunks_.clear();
        //
        LOG_DEBUG_MSG("receiver " << hexpointer(this)
            << "Cleaned up, posting self back to rma stack");
        handler_(this);
    }

    // --------------------------------------------------------------------
    void rma_receiver::handle_error(struct fi_cq_err_entry err)
    {
        LOG_ERROR_MSG("rma_receiver handling an error " << hexpointer(this));
    }

}}}}
