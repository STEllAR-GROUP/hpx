//  Copyright (c) 2015-2017 John Biddiscombe
//  Copyright (c) 2017      Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <plugins/parcelport/libfabric/rma_receiver.hpp>
#include <plugins/parcelport/libfabric/parcelport.hpp>

#include <hpx/runtime/parcelset/parcel_buffer.hpp>
#include <hpx/runtime/parcelset/decode_parcels.hpp>

#include <hpx/util/detail/yield_k.hpp>

namespace hpx {
namespace parcelset {
namespace policies {
namespace libfabric
{
    rma_receiver::rma_receiver(
        parcelport* pp,
        fid_ep* endpoint,
        rdma_memory_pool* memory_pool,
        completion_handler&& handler)
      : pp_(pp),
        endpoint_(endpoint),
        header_region_(nullptr),
        message_region_(nullptr),
        memory_pool_(memory_pool),
        handler_(std::move(handler)),
        rma_count_(0)
    {}

    rma_receiver::rma_receiver(rma_receiver&& other)
      : pp_(other.pp_),
        endpoint_(other.endpoint_),
        header_region_(nullptr),
        message_region_(nullptr),
        memory_pool_(other.memory_pool_),
        handler_(std::move(other.handler_)),
        rma_count_(static_cast<long>(other.rma_count_))
    {
        HPX_ASSERT(other.header_region_ == nullptr);
        HPX_ASSERT(other.message_region_ == nullptr);
    }

    rma_receiver& rma_receiver::operator=(rma_receiver&& other)
    {
        pp_ = other.pp_;
        endpoint_ = other.endpoint_;
        header_region_ = nullptr;
        message_region_ = nullptr;
        memory_pool_ = other.memory_pool_;
        handler_ = std::move(other.handler_);
        rma_count_ = static_cast<long>(other.rma_count_);
        HPX_ASSERT(other.header_region_ == nullptr);
        HPX_ASSERT(other.message_region_ == nullptr);

        return *this;
    }

    rma_receiver::~rma_receiver()
    {
        LOG_DEVEL_MSG("Receiving of message complete " << hexpointer(this));
    }

    void rma_receiver::async_read(
        libfabric_memory_region* region,
        fi_addr_t const& src_addr)
    {
        HPX_ASSERT(rma_count_ == 0);
        HPX_ASSERT(header_region_ == nullptr);
        HPX_ASSERT(message_region_ == nullptr);
        header_region_ = region;

        header_ = reinterpret_cast<header_type*>(header_region_->get_address());
        src_addr_ = src_addr;
        rma_count_ = header_->num_chunks().first;
        HPX_ASSERT(rma_count_ == header_->num_chunks().first);

        HPX_ASSERT(header_region_);
        HPX_ASSERT(header_region_->get_address());
        HPX_ASSERT(header_);
        LOG_DEVEL_MSG( "received " <<
                "buffsize " << decnumber(header_->size())
                << "chunks zerocopy( " << decnumber(header_->num_chunks().first) << ") "
                << ", chunk_flag " << decnumber(header_->header_length())
                << ", normal( " << decnumber(header_->num_chunks().second) << ") "
                << " chunkdata " << decnumber((header_->chunk_data()!=nullptr))
                << " piggyback " << decnumber((header_->piggy_back()!=nullptr))
                << " tag " << hexuint64(header_->tag())
        );

        if (!header_->piggy_back())
        {
            ++rma_count_;
        }

        // If we have no zero copy chunks and piggy backed data, we can
        // process the message immediately, otherwise, dispatch to rma_receiver
        // If we have neither piggy back, nor zero copy chunks, rma_count is 0
        if (rma_count_ == 0)
        {
            handle_non_rma();
            handler_(this);
            return;
        }

        chunks_.resize(header_->num_chunks().first + header_->num_chunks().second);

        char *chunk_data = header_->chunk_data();
        HPX_ASSERT(chunk_data);

        size_t chunkbytes =
            chunks_.size() * sizeof(chunk_struct);

        std::memcpy(chunks_.data(), chunk_data, chunkbytes);
        LOG_DEBUG_MSG("Copied chunk data from header : size "
            << decnumber(chunkbytes));

        LOG_EXCLUSIVE(
        for (const chunk_struct &c : recv_data.chunks)
        {
            LOG_DEBUG_MSG("recv : chunk : size " << hexnumber(c.size_)
                << " type " << decnumber((uint64_t)c.type_)
                << " rkey " << hexpointer(c.rkey_)
                << " cpos " << hexpointer(c.data_.cpos_)
                << " pos " << hexpointer(c.data_.pos_)
                << " index " << decnumber(c.data_.index_));
        });

        rma_regions_.reserve(rma_count_);

        std::size_t index = 0;
        for (const chunk_struct &c : chunks_)
        {
            if (c.type_ == serialization::chunk_type_pointer)
            {
                libfabric_memory_region *get_region =
                    memory_pool_->allocate_region(c.size_);

                LOG_DEVEL_MSG("RDMA Get addr " << hexpointer(c.data_.cpos_)
                    << "rkey " << hexpointer(c.rkey_)
                    << "size " << hexnumber(c.size_)
                    << "tag " << hexuint64(header_->tag())
                    << "local addr " << hexpointer(get_region->get_address())
                    << "length " << hexlength(c.size_));

                rma_regions_.push_back(get_region);

                // overwrite the serialization data to account for the
                // local pointers instead of remote ones
                const void *remoteAddr = c.data_.cpos_;
                chunks_[index] =
                    hpx::serialization::create_pointer_chunk(
                        get_region->get_address(), c.size_, c.rkey_);

                // post the rdma read/get
                LOG_DEVEL_MSG("RDMA Get fi_read :"
                    << " chunk " << decnumber(index)
                    << " client " << hexpointer(endpoint_)
                    << " fi_addr " << hexpointer(src_addr_)
                    << " local addr " << hexpointer(get_region->get_address())
                    << " local desc " << hexpointer(get_region->get_desc())
                    << " size " << hexnumber(c.size_)
                    << " rkey " << hexpointer(c.rkey_)
                    << " remote cpos " << hexpointer(remoteAddr)
                    << " remote pos " << hexpointer(remoteAddr)
                    << " index " << decnumber(c.data_.index_));

                ssize_t ret = fi_read(endpoint_,
                    get_region->get_address(), c.size_, get_region->get_desc(),
                    src_addr_,
                    (uint64_t)(remoteAddr), c.rkey_, this);

                if (ret) throw fabric_error(ret, "fi_read error");
            }
            ++index;
        }
        LOG_DEBUG_MSG("piggy_back is " << hexpointer(piggy_back)
            << " chunk data is " << hexpointer(header_->chunk_data()));

        // If the main message was not piggy backed
        if (!header_->piggy_back())
        {
            std::size_t size = header_->get_message_rdma_size();
            message_region_ = memory_pool_->allocate_region(size);
            message_region_->set_message_length(size);

            LOG_DEVEL_MSG("RDMA Get fi_read message :"
                << " client " << hexpointer(endpoint_)
                << " fi_addr " << hexpointer(src_addr_)
                << " local addr " << hexpointer(message_region_->get_address())
                << " local desc " << hexpointer(message_region_->get_desc())
                << " remote addr " << hexpointer(header_->get_message_rdma_addr())
                << " size " << hexnumber(size)
            );

            ssize_t ret = fi_read(endpoint_,
                message_region_->get_address(), size, message_region_->get_desc(),
                src_addr_,
                (uint64_t)header_->get_message_rdma_addr(),
                header_->get_message_rdma_key(), this);

            if (ret) throw fabric_error(ret, "fi_read error");
        }
    }

    void rma_receiver::handle_non_rma()
    {
        typedef pinned_memory_vector<char, header_size> rcv_data_type;
        typedef parcel_buffer<rcv_data_type, std::vector<char>> rcv_buffer_type;

        LOG_DEVEL_MSG("handle piggy backed sends without zero copy regions");
//         send_ack();

        HPX_ASSERT(header_);
        char *piggy_back = header_->piggy_back();
        HPX_ASSERT(piggy_back);
        rcv_data_type wrapped_pointer(piggy_back, header_->size(), [](){},
                memory_pool_, nullptr);

        rcv_buffer_type buffer(std::move(wrapped_pointer), memory_pool_);
        LOG_DEBUG_MSG("calling parcel decode for complete NORMAL parcel");
        std::size_t num_thread = hpx::get_worker_thread_num();
        parcelset::decode_parcels(*pp_, std::move(buffer), num_thread);
        LOG_DEVEL_MSG("parcel decode called for complete NORMAL parcel");

        memory_pool_->deallocate(header_region_);
        header_region_ = nullptr;
        chunks_.clear();
    }

    void rma_receiver::handle_read_completion()
    {
        // If we haven't read all chunks, we can return and wait
        // for the other incoming read completions
        if (--rma_count_ > 0)
        {
            return;
        }
        send_ack();

        HPX_ASSERT(rma_count_ == 0);


        std::size_t message_length = 0;
        char *message = nullptr;
        if (message_region_)
        {
            message = static_cast<char *>(message_region_->get_address());
            message_length = message_region_->get_message_length();
            LOG_DEBUG_MSG("No piggy_back message, RDMA GET : "
                << hexpointer(message_region_)
                << " length " << decnumber(message_length));
        }
        else
        {
            message_length = header_->size();
            message = header_->piggy_back();
        }

        typedef pinned_memory_vector<char, header_size> rcv_data_type;
        typedef parcel_buffer<rcv_data_type, std::vector<char>> rcv_buffer_type;

        rcv_data_type wrapped_pointer(message, message_length,
            [](){}, memory_pool_, nullptr);
        rcv_buffer_type buffer(std::move(wrapped_pointer), memory_pool_);
        LOG_DEBUG_MSG("calling parcel decode for complete ZEROCOPY parcel");

        LOG_EXCLUSIVE(
        for (chunk_struct &c : chunks_) {
            LOG_DEBUG_MSG("get : chunk : size " << hexnumber(c.size_)
                    << " type " << decnumber((uint64_t)c.type_)
                    << " rkey " << hexpointer(c.rkey_)
                    << " cpos " << hexpointer(c.data_.cpos_)
                    << " pos " << hexpointer(c.data_.pos_)
                    << " index " << decnumber(c.data_.index_));
        })

        buffer.num_chunks_ = header_->num_chunks();
        //buffer.data_.resize(static_cast<std::size_t>(header_->size()));
        //buffer.data_size_ = header_->size();
        buffer.chunks_.resize(chunks_.size());
        decode_message_with_chunks(*pp_, std::move(buffer), 0, chunks_);
        LOG_DEVEL_MSG("parcel decode called for ZEROCOPY complete parcel");

        for (auto region: rma_regions_)
            memory_pool_->deallocate(region);
        rma_regions_.clear();
        chunks_.clear();

        memory_pool_->deallocate(header_region_);
        header_region_ = nullptr;
        if (message_region_)
        {
            memory_pool_->deallocate(message_region_);
            message_region_ = nullptr;
        }
        handler_(this);
    }

    void rma_receiver::send_ack()
    {

#if HPX_PARCELPORT_LIBFABRIC_IMM_UNSUPPORTED
        LOG_DEVEL_MSG("RDMA Get tag " << hexuint64(header_->tag())
            << " has completed : posting 8 byte ack to origin");


        int ret = 0;
        for (std::size_t k = 0; true; ++k)
        {
            std::uint64_t tag = header_->tag();
            ret = fi_inject(endpoint_, &tag, sizeof(std::uint64_t), src_addr_);
            if (ret == -FI_EAGAIN)
            {
                LOG_DEVEL_MSG("reposting inject...\n");
                hpx::util::detail::yield_k(k,
                    "libfabric::rma_receiver::handle_read_completion");
                continue;
            }
            if (ret) throw fabric_error(ret, "fi_inject ack notification error");
            break;
        }

#else
        LOG_DEVEL_MSG("RDMA Get tag " << hexuint64(header_->tag())
            << " has completed : posting zero byte ack to origin");
        std::terminate();
#endif
    }
}}}}
