//  Copyright (c) 2015 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// config
#include <hpx/config/defines.hpp>

// util
#include <hpx/util/command_line_handling.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/util/memory_chunk_pool_allocator.hpp>
#include <hpx/lcos/local/condition_variable.hpp>

#include <hpx/apply.hpp>

// The memory pool specialization need to be pulled in before encode_parcels
#include "RdmaMemoryPool.h"

// parcelport main
#include <hpx/runtime.hpp>
#include <hpx/runtime/parcelset/parcelport.hpp>
#include <hpx/runtime/parcelset/parcel_buffer.hpp>
#include <hpx/runtime/parcelset/encode_parcels.hpp>
#include <hpx/runtime/parcelset/decode_parcels.hpp>
#include <hpx/plugins/parcelport_factory.hpp>
#include <hpx/runtime/parcelset/parcelport_impl.hpp>
#include <hpx/runtime/serialization/detail/future_await_container.hpp>

// Local parcelport plugin
// #define USE_SPECIALIZED_SCHEDULER
#include "sender_connection.hpp"
#include "connection_handler.hpp"
#include "locality.hpp"
#include "header.hpp"
#include "pinned_memory_vector.hpp"
#ifdef USE_SPECIALIZED_SCHEDULER
# include "scheduler.hpp"
#endif
//
// rdmahelper library
#include "RdmaLogging.h"
#include "RdmaController.h"
#include "RdmaDevice.h"
//
#include <unordered_map>
//
#include <memory>
#include <mutex>
#include <condition_variable>

#define HPX_PARCELPORT_VERBS_MEMORY_COPY_THRESHOLD RDMA_DEFAULT_MEMORY_POOL_SMALL_CHUNK_SIZE
#define HPX_PARCELPORT_VERBS_MAX_SEND_QUEUE        32

// Note HPX_PARCELPORT_VERBS_IMM_UNSUPPORTED is set by CMake configuration
// if the machine is a BlueGene active storage node which does not support immediate
// data sends

using namespace hpx::parcelset::policies;

namespace hpx { namespace parcelset {
    namespace policies { namespace verbs
    {

    // --------------------------------------------------------------------
    // simple atomic counter we use for tags
    // when a parcel is sent to a remote locality, it may need to pull zero copy chunks from us.
    // we keep the chunks until the remote locality sends a zero byte message with the tag we gave
    // them and then we know it is safe to release the memory back to the pool.
    // The tags can have a short lifetime, but must be unique, so we encode the ip address with
    // a counter to generate tags per destination.
    // The tag is sent in immediate data so must be 32bits only : Note that the tag only has a
    // lifetime of the unprocessed parcel, so it can be reused as soon as the parcel has been completed
    // and therefore a 16bit count is sufficient as we only keep a few parcels per locality in flight at a time
    // --------------------------------------------------------------------
    struct tag_provider {
        tag_provider() : next_tag_(1) {}

        uint32_t next(uint32_t ip_addr)
        {
            // @TODO track wrap around and collisions (how?)
            return (next_tag_++ & 0x0000FFFF) + (ip_addr << 16);
        }

        // using 16 bits currently.
        boost::atomic<uint32_t> next_tag_;
    };

    // --------------------------------------------------------------------
    // parcelport, the implementation of the parcelport itself
    // --------------------------------------------------------------------
    class HPX_EXPORT parcelport
      : public parcelport_impl<parcelport>
    {
    private:
        typedef parcelport_impl<parcelport> base_type;

        static std::size_t max_connections(util::runtime_configuration const& ini)
        {
            LOG_DEBUG_MSG("This should not be necessary as the parcelport_impl does it for us");
            return hpx::util::get_entry_as<std::size_t>(
                // this uses the mpi value as we have not bothered to add a verbs one yet
                ini, "hpx.parcel.mpi.max_connections", HPX_PARCEL_MAX_CONNECTIONS);
        }

        // --------------------------------------------------------------------
        // returns a locality object that represents 'this' locality
        // --------------------------------------------------------------------
        static parcelset::locality here(util::runtime_configuration const& ini)
        {
            FUNC_START_DEBUG_MSG;
            if (ini.has_section("hpx.parcel.verbs")) {
                util::section const * sec = ini.get_section("hpx.parcel.verbs");
                if (NULL != sec) {
                    std::string ibverbs_enabled(sec->get_entry("enable", "0"));
                    if (boost::lexical_cast<int>(ibverbs_enabled)) {
                        // _ibverbs_ifname    = sec->get_entry("ifname",    HPX_PARCELPORT_VERBS_IFNAME);
                        _ibverbs_device    = sec->get_entry("device",    HPX_PARCELPORT_VERBS_DEVICE);
                        _ibverbs_interface = sec->get_entry("interface", HPX_PARCELPORT_VERBS_INTERFACE);
                        char buff[256];
                        _ibv_ip = hpx::parcelset::policies::verbs::Get_rdma_device_address(_ibverbs_device.c_str(), _ibverbs_interface.c_str(), buff);
                        LOG_DEBUG_MSG("here() got hostname of " << buff);
                    }
                }
            }
            if (ini.has_section("hpx.agas")) {
                util::section const* sec = ini.get_section("hpx.agas");
                if (NULL != sec) {
                    LOG_DEBUG_MSG("hpx.agas port number " << decnumber(hpx::util::get_entry_as<boost::uint16_t>(*sec, "port", HPX_INITIAL_IP_PORT)));
                    _port = hpx::util::get_entry_as<boost::uint16_t>(*sec, "port", HPX_INITIAL_IP_PORT);
                }
            }
            if (ini.has_section("hpx.parcel")) {
                util::section const* sec = ini.get_section("hpx.parcel");
                if (NULL != sec) {
                    LOG_DEBUG_MSG("hpx.parcel port number " << hpx::util::get_entry_as<boost::uint16_t>(*sec, "port", HPX_INITIAL_IP_PORT));
                }
            }
            FUNC_END_DEBUG_MSG;
            return parcelset::locality(locality(_ibv_ip));
        }

    public:
        // --------------------------------------------------------------------
        // main vars used to manage the RDMA controller and interface
        // --------------------------------------------------------------------
        static RdmaControllerPtr _rdmaController;
        // static std::string       _ibverbs_ifname;
        static std::string       _ibverbs_device;
        static std::string       _ibverbs_interface;
        static boost::uint32_t   _port;
        static boost::uint32_t   _ibv_ip;
        // to quickly lookup a que-pair (QP) from a destination ip address
        typedef std::map<boost::uint32_t, boost::uint32_t> ip_map;
        typedef ip_map::iterator                           ip_map_iterator;
        //
        ip_map ip_qp_map;

        // @TODO, clean up the allocators, buffers, chunk_pool etc so that there is a more consistent
        // reuse of classes/types. the use of pointer allocators etc is a dreadful hack and
        // needs reworking
        typedef header<RDMA_DEFAULT_MEMORY_POOL_SMALL_CHUNK_SIZE>  header_type;
        typedef hpx::lcos::local::spinlock                   mutex_type;
        typedef std::lock_guard<mutex_type>                  scoped_lock;
        typedef hpx::lcos::local::condition_variable_any     condition_type;
        typedef std::unique_lock<mutex_type>                 unique_lock;

        // note use std::mutex in stop function as HPX is terminating
        mutex_type  stop_mutex;
        mutex_type  connection_mutex;
        mutex_type  ReadCompletionMap_mutex;
        mutex_type  SendCompletionMap_mutex;
        mutex_type  TagSendCompletionMap_mutex;

        typedef RdmaMemoryPool                                    memory_pool_type;
        typedef std::shared_ptr<memory_pool_type>                 memory_pool_ptr_type;
        typedef hpx::util::detail::memory_chunk_pool_allocator
                <char, memory_pool_type, mutex_type>               allocator_type;
        typedef util::detail::pinned_memory_vector<char>           rcv_data_type;
        typedef parcel_buffer<rcv_data_type>                       snd_buffer_type;
        typedef parcel_buffer<rcv_data_type, std::vector<char>>    rcv_buffer_type;

        //
        boost::atomic<bool>       stopped_;
        boost::atomic_uint        active_send_count_;
        memory_pool_ptr_type      chunk_pool_;
        verbs::tag_provider       tag_provider_;
        std::atomic_flag          connection_started;

#ifdef USE_SPECIALIZED_SCHEDULER
        custom_scheduler          parcelport_scheduler;
#endif
        // performance_counters::parcels::gatherer& parcels_sent_;

        // --------------------------------------------------------------------
        // struct we use to keep track of all memory regions used during a send, they must
        // be held onto until all transfers of data are complete.
        // --------------------------------------------------------------------
        typedef struct parcel_send_data_ {
            uint32_t                                         tag;
            std::atomic_flag                                 delete_flag;
            bool                                             has_zero_copy;
            util::unique_function_nonser< void(error_code const&) > handler;

            RdmaMemoryRegion *header_region, *chunk_region, *message_region;
            std::vector<RdmaMemoryRegion*>                   zero_copy_regions;
        } parcel_send_data;

        // --------------------------------------------------------------------
        // struct we use to keep track of all memory regions used during a recv, they must
        // be held onto until all transfers of data are complete.
        // --------------------------------------------------------------------
        typedef struct {
            std::atomic_uint                                 rdma_count;
            uint32_t                                         tag;
            std::vector<serialization::serialization_chunk>  chunks;
            RdmaMemoryRegion *header_region, *chunk_region, *message_region;
            std::vector<RdmaMemoryRegion*>                   zero_copy_regions;
        } parcel_recv_data;

        typedef std::list<parcel_send_data>      active_send_list_type;
        typedef active_send_list_type::iterator  active_send_iterator;
        //
        typedef std::list<parcel_recv_data>      active_recv_list_type;
        typedef active_recv_list_type::iterator  active_recv_iterator;

        // map send/recv parcel wr_id to all info needed on completion
        typedef std::unordered_map<uint64_t, active_send_iterator> send_wr_map;
        typedef std::unordered_map<uint64_t, active_recv_iterator> recv_wr_map;

        // store received objects using a map referenced by verbs work request ID
        send_wr_map SendCompletionMap;
        send_wr_map TagSendCompletionMap;

        recv_wr_map ReadCompletionMap;

        active_send_list_type   active_sends;
        mutex_type              active_send_mutex;
        condition_type          active_send_condition;

        active_recv_list_type active_recvs;
        mutex_type       active_recv_mutex;
        std::atomic<int> total_receives;

        // --------------------------------------------------------------------
        // Constructor : mostly just initializes the superclass with 'here'
        // --------------------------------------------------------------------
        parcelport(util::runtime_configuration const& ini,
                util::function_nonser<void(std::size_t, char const*)> const& on_start_thread,
                util::function_nonser<void()> const& on_stop_thread)
              : base_type(ini, here(ini), on_start_thread, on_stop_thread)
              , stopped_(false)
              , active_send_count_(0)
              , connection_started(ATOMIC_FLAG_INIT)
              , total_receives(0)
              // , parcels_sent_(0)

        {
            FUNC_START_DEBUG_MSG;
            // we need this for background OS threads to get 'this' pointer
            parcelport::_parcelport_instance = this;
            // port number is set during locality initialization in 'here()'
            _rdmaController = std::make_shared<RdmaController>
                (_ibverbs_device.c_str(), _ibverbs_interface.c_str(), _port);
            //
            connection_started.clear();
            FUNC_END_DEBUG_MSG;
        }

        void io_service_work()
        {
            // We only execute work on the IO service while HPX is starting
            while (hpx::is_starting())
            {
                OS_background_work();
//                LOG_TRACE_MSG("OS background work");
            }
        }


        // Start the handling of connections.
        bool do_run()
        {
            FUNC_START_DEBUG_MSG;

            _rdmaController->startup();

            LOG_DEBUG_MSG("Fetching memory pool");
            chunk_pool_ = _rdmaController->getMemoryPool();

            LOG_DEBUG_MSG("Setting Pre-Connection function");
            auto preConnection_function = std::bind( &parcelport::handle_verbs_preconnection, this);
            _rdmaController->setPreConnectionFunction(preConnection_function);

            LOG_DEBUG_MSG("Setting Connection function");
            auto connection_function = std::bind( &parcelport::handle_verbs_connection, this, std::placeholders::_1, std::placeholders::_2);
            _rdmaController->setConnectionFunction(connection_function);

            LOG_DEBUG_MSG("Setting Completion function");
            // need to use std:bind here as rdmahelper lib uses it too
            auto completion_function = std::bind( &parcelport::handle_verbs_completion, this, std::placeholders::_1, std::placeholders::_2);
            _rdmaController->setCompletionFunction(completion_function);

            for (std::size_t i = 0; i != io_service_pool_.size(); ++i)
            {
                io_service_pool_.get_io_service(int(i)).post(
                    hpx::util::bind(
                        &parcelport::io_service_work, this
                    )
                );
            }

#ifdef USE_SPECIALIZED_SCHEDULER
            // initialize our custom scheduler
            parcelport_scheduler.init();

            //
            hpx::error_code ec(hpx::lightweight);
            parcelport_scheduler.register_thread_nullary(
                    util::bind(&parcelport::hpx_background_work_thread, this),
                    "hpx_background_work_thread",
                    threads::pending, true, threads::thread_priority_critical,
                    std::size_t(-1), threads::thread_stacksize_default, ec);

            FUNC_END_DEBUG_MSG;
            return ec ? false : true;
#else
            return true;
#endif
       }

        // --------------------------------------------------------------------
        //  return a sender_connection object back to the parcelport_impl
        // --------------------------------------------------------------------
        std::shared_ptr<sender_connection> create_connection(
            parcelset::locality const& dest, error_code& ec)
        {
            FUNC_START_DEBUG_MSG;

            boost::uint32_t dest_ip = dest.get<locality>().ip_;
            LOG_DEBUG_MSG("Locality " << ipaddress(_ibv_ip) << " create_connection to " << ipaddress(dest_ip) );

            RdmaClient *client = get_remote_connection(dest);
            std::shared_ptr<sender_connection> result = std::make_shared<sender_connection>(
                  this
                , dest_ip
                , dest.get<locality>()
                , client
                , chunk_pool_.get()
                , parcels_sent_
            );

            FUNC_END_DEBUG_MSG;
            return result;
        }

        // ----------------------------------------------------------------------------------------------
        // Clean up a completed send and all its regions etc
        // Called when we finish sending a simple message, or when all zero-copy Get operations are done
        // ----------------------------------------------------------------------------------------------
        void delete_send_data(active_send_iterator send) {
            parcel_send_data &send_data = *send;
            // trigger the send complete handler for hpx internal cleanup
            LOG_DEBUG_MSG("Calling write_handler for completed send");
//            send_data.handler.operator()(error_code(), send_data.parcel);
            send_data.handler.operator()(error_code());
            //
            LOG_DEBUG_MSG("deallocating region 1 for completed send " << hexpointer(send_data.header_region));
            chunk_pool_->deallocate(send_data.header_region);
            send_data.header_region  = NULL;
            // if this message had multiple (2) SGEs then release other regions
            if (send_data.message_region) {
                LOG_DEBUG_MSG("deallocating region 2 for completed send " << hexpointer(send_data.message_region));
                chunk_pool_->deallocate(send_data.message_region);
                send_data.message_region = NULL;
            }
            for (auto r : send_data.zero_copy_regions) {
                LOG_DEBUG_MSG("Deallocating " << hexpointer(r));
                chunk_pool_->deallocate(r);
            }
            //
            // when a parcel is deleted, it takes a lock, since we are locking before delete
            // we must grab a reference to the parcel and keep it alive until we unlock
            // parcelset::parcel parcel = std::move(send->parcel);
            // erratum : the parcel destructor takes a lock even when empty, so better
            // to avoid the lock held detection by using util::ignore_while_checking
            {
                unique_lock lock(active_send_mutex);
//                util::ignore_while_checking<unique_lock> il(&lock);
                active_sends.erase(send);
                --active_send_count_;
                LOG_DEBUG_MSG("Active send after erase size " << hexnumber(active_send_count_) );
                lock.unlock();
                active_send_condition.notify_one();
            }
        }

        // ----------------------------------------------------------------------------------------------
        // Clean up a completed recv and all its regions etc
        // Called when a parcel_buffer finishes decoding a message
        // ----------------------------------------------------------------------------------------------
        void delete_recv_data(active_recv_iterator recv)
        {
            FUNC_START_DEBUG_MSG;
            parcel_recv_data &recv_data = *recv;

            chunk_pool_->deallocate(recv_data.header_region);
            LOG_DEBUG_MSG("Zero copy regions size is (delete) " << decnumber(recv_data.zero_copy_regions.size()));
            for (auto r : recv_data.zero_copy_regions) {
                LOG_DEBUG_MSG("Deallocating " << hexpointer(r));
                chunk_pool_->deallocate(r);
            }
            {
                scoped_lock lock(active_recv_mutex);
                active_recvs.erase(recv);
                LOG_DEBUG_MSG("Active recv after erase size " << hexnumber(active_recvs.size()) );
            }
        }

        int handle_verbs_preconnection() {
            if (connection_started.test_and_set(std::memory_order_acquire)) {
                LOG_ERROR_MSG("Got a connection request during race detection");
                return 0;
            }
            return 1;
        }

        // ----------------------------------------------------------------------------------------------
        // handler for connections, this is triggered as a callback from the rdmaController when
        // a connection event has occurred.
        // When we connect to another locality, all internal structures are updated accordingly,
        // but when another locality connects to us, we must do it manually via this callback
        // ----------------------------------------------------------------------------------------------
        int handle_verbs_connection(std::pair<uint32_t,uint64_t> qpinfo, RdmaClientPtr client)
        {
            scoped_lock lock(connection_mutex);
            LOG_DEBUG_MSG("handle_verbs_connection callback triggered");
            boost::uint32_t dest_ip = client->getRemoteIPv4Address();
            ip_map_iterator ip_it = ip_qp_map.find(dest_ip);
            if (ip_it==ip_qp_map.end()) {
                ip_qp_map.insert(std::make_pair(dest_ip, qpinfo.first));
                LOG_DEBUG_MSG("handle_verbs_connection OK adding " << ipaddress(dest_ip));
            }
            else {
                throw std::runtime_error("verbs parcelport : should not be receiving a connection more than once");
            }
            return 0;
        }

        RdmaMemoryRegion *getRdmaRegion(std::size_t size) {
            if (size<=RDMA_DEFAULT_MEMORY_POOL_LARGE_CHUNK_SIZE) {
                return chunk_pool_->allocateRegion(size);
            }
            return chunk_pool_->AllocateTemporaryBlock(size);
        }
/*
        void postRDMAGet(std::size_t size, void const *pos, boost::uint32_t rkey, active_recv_iterator recv_data, serialization::serialization_chunk &chunk, RdmaClient *client) {
            RdmaMemoryRegion *get_region = getRdmaRegion(size);
            LOG_DEBUG_MSG("RDMA Get address " << hexpointer(pos)
                    << " rkey " << decnumber(rkey) << " size " << hexnumber(size)
                    << " tag " << hexuint32(recv_data->tag)
                    << " local address " << get_region->getAddress() << " length " << size);
            recv_data->zero_copy_regions.push_back(get_region);
            LOG_DEBUG_MSG("Zero copy regions size is (create) " << decnumber(recv_data->zero_copy_regions.size()));
            // put region into map before posting read in case it completes whilst this thread is suspended
            {
                scoped_lock lock(ReadCompletionMap_mutex);
                ReadCompletionMap[(uint64_t)get_region] = recv_data;
            }
            // overwrite the serialization data to account for the local pointers instead of remote ones
            /// post the rdma read/get
            const void *remoteAddr = pos;
            chunk = hpx::serialization::create_pointer_chunk(get_region->getAddress(), size, rkey);
            client->postRead(get_region, rkey, remoteAddr, size);
        }
*/
        // ----------------------------------------------------------------------------------------------
        // When a send completes take one of two actions ...
        // if there are no zero copy chunks, we consider the parcel sending complete
        // so release memory and trigger write_handler.
        // If there are zero copy chunks to be RDMA GET from the remote end, then
        // we hold onto data until the have completed.
        // ----------------------------------------------------------------------------------------------
        void handle_send_completion(uint64_t wr_id)
        {
            bool                 found_wr_id;
            active_send_iterator current_send;
            {
                // we must be very careful here.
                // if the lock is not obtained and this thread is waiting, then
                // zero copy Gets might complete and another thread might receive a zero copy
                // complete message then delete the current send data whilst we are still waiting
                unique_lock lock(SendCompletionMap_mutex);
                send_wr_map::iterator it = SendCompletionMap.find(wr_id);
                found_wr_id = (it != SendCompletionMap.end());
                if (found_wr_id) {
                    current_send = it->second;
                    LOG_DEBUG_MSG("erasing " << hexpointer(wr_id) << "from SendCompletionMap : size before erase " << SendCompletionMap.size());
                    SendCompletionMap.erase(it);
                }
                else {
#ifdef HPX_PARCELPORT_VERBS_IMM_UNSUPPORTED
                    lock.unlock();
                    handle_tag_send_completion(wr_id);
                    return;
#else
                    LOG_ERROR_MSG("FATAL : SendCompletionMap did not find " << hexpointer(wr_id));
                    // for (auto & pair : SendCompletionMap) {
                    //     std::cout << hexpointer(pair.first) << "\n";
                    // }
                    std::terminate();
#endif
                }
            }
            if (found_wr_id) {
                // if the send had no zero_copy regions, then it has completed
                if (!current_send->has_zero_copy) {
                    LOG_DEBUG_MSG("Deleting send data " << hexpointer(&(*current_send)) << "normal");
                    delete_send_data(current_send);
                }
                // if another thread signals to say zero-copy is complete, delete the data
                else if (current_send->delete_flag.test_and_set(std::memory_order_acquire)) {
                    LOG_DEBUG_MSG("Deleting send data " << hexpointer(&(*current_send)) << "after race detection");
                    delete_send_data(current_send);
                }
            }
            else {
                throw std::runtime_error("RDMA Send completed with unmatched Id");
            }
        }

        // ----------------------------------------------------------------------------------------------
        // When a recv completes, take one of two actions ...
        // if there are no zero copy chunks, we consider the parcel receive complete
        // so release memory and trigger write_handler.
        // If there are zero copy chunks to be RDMA GET from the remote end, then
        // we hold onto data until they have completed.
        // ----------------------------------------------------------------------------------------------
        void handle_recv_completion(uint64_t wr_id, RdmaClient* client)
        {
            util::high_resolution_timer timer;

            // bookkeeping : decrement counter that keeps preposted queue full
            client->popReceive();
            _rdmaController->refill_client_receives();

            // store details about this parcel so that all memory buffers can be kept
            // until all recv operations have completed.
            active_recv_iterator current_recv;
            {
                scoped_lock lock(active_recv_mutex);
                active_recvs.emplace_back();
                current_recv = std::prev(active_recvs.end());
                LOG_DEBUG_MSG("Active recv after insert size " << hexnumber(active_recvs.size()));
            }
            parcel_recv_data &recv_data = *current_recv;
            // get the header of the new message/parcel
            recv_data.header_region  = (RdmaMemoryRegion *)wr_id;
            header_type *h = (header_type*)recv_data.header_region->getAddress();
            recv_data.message_region = NULL;
            recv_data.chunk_region   = NULL;
            // zero copy chunks we have to GET from the source locality
            if (h->piggy_back()) {
              recv_data.rdma_count        = h->num_chunks().first;
            }
            else {
                recv_data.rdma_count      = 1 + h->num_chunks().first;
            }
            // each parcel has a unique tag which we use to organize zero-copy data if we need any
            recv_data.tag            = h->tag();

            LOG_DEBUG_MSG( "received IBV_WC_RECV " <<
                    "buffsize " << decnumber(h->size())
                    << "numbytes " << decnumber(h->numbytes())
                    << "chunks zerocopy( " << decnumber(h->num_chunks().first) << ") "
                    << ", chunk_flag " << decnumber(h->header_length())
                    << ", normal( " << decnumber(h->num_chunks().second) << ") "
                    << " chunkdata " << decnumber((h->chunk_data()!=NULL))
                    << " piggyback " << decnumber((h->piggy_back()!=NULL))
                    << " tag " << hexuint32(h->tag())
                    << " total receives " << decnumber(++total_receives)
            );

            // setting this flag to false - if more data is needed - disables final parcel receive call
            bool parcel_complete = true;

            // if message was not piggybacked
            char *piggy_back = h->piggy_back();
            char *chunk_data = h->chunk_data();
            if (chunk_data) {
                // all the info about chunks we need is stored inside the header
                recv_data.chunks.resize(h->num_chunks().first + h->num_chunks().second);
                size_t chunkbytes = recv_data.chunks.size() * sizeof(serialization::serialization_chunk);
                std::memcpy(recv_data.chunks.data(), chunk_data, chunkbytes);
                LOG_DEBUG_MSG("Copied chunk data from header : size " << decnumber(chunkbytes));

                // setup info for zero-copy rdma get chunks (if there are any)
                if (recv_data.rdma_count>0) {
                    parcel_complete = false;
                    int index = 0;
                    for (serialization::serialization_chunk &c : recv_data.chunks) {
                        LOG_DEBUG_MSG("recv : chunk : size " << hexnumber(c.size_)
                                << " type " << decnumber((uint64_t)c.type_)
                                << " rkey " << decnumber(c.rkey_)
                                << " cpos " << hexpointer(c.data_.cpos_)
                                << " pos " << hexpointer(c.data_.pos_)
                                << " index " << decnumber(c.data_.index_));
                    }
                    for (serialization::serialization_chunk &c : recv_data.chunks) {
                        if (c.type_ == serialization::chunk_type_pointer) {
                            RdmaMemoryRegion *get_region = getRdmaRegion(c.size_);
                            LOG_DEBUG_MSG("RDMA Get address " << hexpointer(c.data_.cpos_)
                                    << " rkey " << decnumber(c.rkey_) << " size " << hexnumber(c.size_)
                                    << " tag " << hexuint32(recv_data.tag)
                                    << " local address " << get_region->getAddress() << " length " << c.size_);
                            recv_data.zero_copy_regions.push_back(get_region);
                            LOG_DEBUG_MSG("Zero copy regions size is (create) " << decnumber(recv_data.zero_copy_regions.size()));
                            // put region into map before posting read in case it completes whilst this thread is suspended
                            {
                                scoped_lock lock(ReadCompletionMap_mutex);
                                ReadCompletionMap[(uint64_t)get_region] = current_recv;
                            }
                            // overwrite the serialization data to account for the local pointers instead of remote ones
                            /// post the rdma read/get
                            const void *remoteAddr = c.data_.cpos_;
                            recv_data.chunks[index] = hpx::serialization::create_pointer_chunk(get_region->getAddress(), c.size_, c.rkey_);
                            client->postRead(get_region, c.rkey_, remoteAddr, c.size_);
//                            postRDMAGet(c.size_, c.data_.cpos_, c.rkey_, current_recv, recv_data.chunks[index], client);
                        }
                        index++;
                    }
                }
            }
            else {
                LOG_ERROR_MSG("@TODO implement RDMA GET of mass chunk information when header too small");
                std::terminate();
                throw std::runtime_error("@TODO implement RDMA GET of mass chunk information when header too small");
            }

            LOG_DEBUG_MSG("piggy_back is " << hexpointer(piggy_back) << " chunk data is " << hexpointer(h->chunk_data()));
            // if the main serialization chunk is piggybacked in second SGE
            if (piggy_back) {
                if (parcel_complete) {
                    rcv_data_type wrapped_pointer(piggy_back, h->size(),
                        boost::bind(&parcelport::delete_recv_data, this, current_recv), chunk_pool_.get(), NULL);
                    rcv_buffer_type buffer(std::move(wrapped_pointer), chunk_pool_.get());
                    LOG_DEBUG_MSG("calling parcel decode for complete NORMAL parcel");
                    hpx::parcelset::decode_message_with_chunks<parcelport, rcv_buffer_type>
                        (*this, std::move(buffer), 0, recv_data.chunks);
                    LOG_DEBUG_MSG("parcel decode called for complete NORMAL parcel");
                }
            }
            else {
                std::size_t size = h->GetRdmaMessageLength();
                RdmaMemoryRegion *get_region = getRdmaRegion(size);
                get_region->setMessageLength(size);
                recv_data.zero_copy_regions.push_back(get_region);
                // put region into map before posting read in case it completes whilst this thread is suspended
                {
                    scoped_lock lock(ReadCompletionMap_mutex);
                    ReadCompletionMap[(uint64_t)get_region] = current_recv;
                }
                const void *remoteAddr = h->GetRdmaAddr();
                LOG_DEBUG_MSG("@TODO Pushing back an extra chunk description");
                recv_data.chunks.push_back(
                    hpx::serialization::create_pointer_chunk(get_region->getAddress(), size, h->GetRdmaKey()));
                client->postRead(get_region, h->GetRdmaKey(), remoteAddr, h->GetRdmaMessageLength());
            }

            // @TODO replace performance counter data
            //          performance_counters::parcels::data_point& data = buffer.data_point_;
            //          data.time_ = timer.elapsed_nanoseconds();
            //          data.bytes_ = static_cast<std::size_t>(buffer.size_);
            //          ...
            //          data.time_ = timer.elapsed_nanoseconds() - data.time_;
        }

#ifdef HPX_PARCELPORT_VERBS_IMM_UNSUPPORTED
        void handle_tag_send_completion(uint64_t wr_id)
        {
            LOG_DEBUG_MSG("Handle 4 byte completion" << hexpointer(wr_id));
            RdmaMemoryRegion *region = (RdmaMemoryRegion *)wr_id;
            uint32_t tag = *(uint32_t*) (region->getAddress());
            chunk_pool_->deallocate(region);
            LOG_DEBUG_MSG("Cleaned up from 4 byte ack message with tag " << hexuint32(tag));
        }
#endif

        void handle_tag_recv_completion(uint64_t wr_id, uint32_t tag, const RdmaClient *client)
        {
#ifdef HPX_PARCELPORT_VERBS_IMM_UNSUPPORTED
            RdmaMemoryRegion *region = (RdmaMemoryRegion *)wr_id;
            tag = *((uint32_t*) (region->getAddress()));
            LOG_DEBUG_MSG("Received 4 byte ack message with tag " << hexuint32(tag));
#else
            RdmaMemoryRegion *region = (RdmaMemoryRegion *)wr_id;
            LOG_DEBUG_MSG("Received 0 byte ack message with tag " << hexuint32(tag));
#endif
            // bookkeeping : decrement counter that keeps preposted queue full
            client->popReceive();

            // let go of this region (waste really as this was a zero byte message)
            chunk_pool_->deallocate(region);

            // now release any zero copy regions we were holding until parcel complete
            active_send_iterator current_send;
            {
                scoped_lock lock(TagSendCompletionMap_mutex);
                send_wr_map::iterator it = TagSendCompletionMap.find(tag);
                if (it==TagSendCompletionMap.end()) {
                    LOG_ERROR_MSG("Tag not present in Send map, FATAL");
                    std::terminate();
                }
                current_send = it->second;
                TagSendCompletionMap.erase(it);
            }
            // we cannot delete the send data until we are absolutely sure that
            // the initial send has been cleaned up
            if (current_send->delete_flag.test_and_set(std::memory_order_acquire)) {
                LOG_DEBUG_MSG("Deleting send data " << hexpointer(&(*current_send)) << "with no race detection");
                delete_send_data(current_send);
            }
            //
            _rdmaController->refill_client_receives();

            LOG_DEBUG_MSG( "received IBV_WC_RECV handle_tag_recv_completion "
                    << " total receives " << decnumber(++total_receives)
            );

        }

        void handle_rdma_get_completion(uint64_t wr_id, RdmaClient *client)
        {
            bool                 found_wr_id;
            active_recv_iterator current_recv;
            {   // locked region : make sure map isn't modified whilst we are querying it
                scoped_lock lock(ReadCompletionMap_mutex);
                recv_wr_map::iterator it = ReadCompletionMap.find(wr_id);
                found_wr_id = (it != ReadCompletionMap.end());
                if (found_wr_id) {
                    current_recv = it->second;
                    LOG_DEBUG_MSG("erasing " << hexpointer(wr_id) << "from ReadCompletionMap : size before erase " << ReadCompletionMap.size());
                    ReadCompletionMap.erase(it);
                }
                else {
                    LOG_ERROR_MSG("Fatal error as wr_id is not in completion map");
                    std::terminate();
                }
            }
            if (found_wr_id) {
                parcel_recv_data &recv_data = *current_recv;
                LOG_DEBUG_MSG("RDMA Get tag " << hexuint32(recv_data.tag) << " has count of " << decnumber(recv_data.rdma_count));
                if (--recv_data.rdma_count > 0) {
                    // we can't do anything until all zero copy chunks are here
                    return;
                }
                //
#ifdef HPX_PARCELPORT_VERBS_IMM_UNSUPPORTED
                LOG_DEBUG_MSG("RDMA Get tag " << hexuint32(recv_data.tag) << " has completed : posting 4 byte ack to origin");
                RdmaMemoryRegion *tag_region = chunk_pool_->allocateRegion(4); // space for a uint32_t
                uint32_t *tag_memory = (uint32_t*)(tag_region->getAddress());
                *tag_memory = recv_data.tag;
                tag_region->setMessageLength(4);
                client->postSend(tag_region, true, false, 0);
#else
                LOG_DEBUG_MSG("RDMA Get tag " << hexuint32(recv_data.tag) << " has completed : posting zero byte ack to origin");
                // convert uint32 to uint64 so we can use it as a fake message region (wr_id only for 0 byte send)
                uint64_t fake_region = recv_data.tag;
                client->postSend_x0((RdmaMemoryRegion*)fake_region, false, true, recv_data.tag);
#endif
                //
                LOG_DEBUG_MSG("Zero copy regions size is (completion) " << decnumber(recv_data.zero_copy_regions.size()));

                header_type *h = (header_type*)recv_data.header_region->getAddress();
                LOG_DEBUG_MSG( "get completion " <<
                        "buffsize " << decnumber(h->size())
                        << "numbytes " << decnumber(h->numbytes())
                        << "chunks zerocopy( " << decnumber(h->num_chunks().first) << ") "
                        << ", chunk_flag " << decnumber(h->header_length())
                        << ", normal( " << decnumber(h->num_chunks().second) << ") "
                        << " chunkdata " << decnumber((h->chunk_data()!=NULL))
                        << " piggyback " << decnumber((h->piggy_back()!=NULL))
                        << " tag " << hexuint32(h->tag())
                );

                std::size_t message_length;
                char *message = h->piggy_back();
                if (message) {
                    message_length = h->size();
                }
                else {
                    RdmaMemoryRegion *message_region = recv_data.zero_copy_regions.back();
                    recv_data.zero_copy_regions.resize(recv_data.zero_copy_regions.size()-1);
                    message = static_cast<char *>(message_region->getAddress());
                    message_length = message_region->getMessageLength();
                    LOG_DEBUG_MSG("No piggy_back message, RDMA GET : " << hexpointer(message_region) << " length " decnumber(message_length));
                    LOG_DEBUG_MSG("No piggy_back message, RDMA GET : " << hexpointer(recv_data.message_region) << " length " decnumber(message_length));
                }

                LOG_DEBUG_MSG("Creating a release buffer callback for tag " << hexuint32(recv_data.tag));
                rcv_data_type wrapped_pointer(message, message_length,
                        boost::bind(&parcelport::delete_recv_data, this, current_recv),
                        chunk_pool_.get(), NULL);
                rcv_buffer_type buffer(std::move(wrapped_pointer), chunk_pool_.get());
                LOG_DEBUG_MSG("calling parcel decode for complete ZEROCOPY parcel");

                for (serialization::serialization_chunk &c : recv_data.chunks) {
                    LOG_DEBUG_MSG("get : chunk : size " << hexnumber(c.size_)
                            << " type " << decnumber((uint64_t)c.type_)
                            << " rkey " << decnumber(c.rkey_)
                            << " cpos " << hexpointer(c.data_.cpos_)
                            << " pos " << hexpointer(c.data_.pos_)
                            << " index " << decnumber(c.data_.index_));
                }

                buffer.num_chunks_ = h->num_chunks();
                //buffer.data_.resize(static_cast<std::size_t>(h->size()));
                //buffer.data_size_ = h->size();
                buffer.chunks_.resize(recv_data.chunks.size());
                decode_message_with_chunks(*this, std::move(buffer), 0, recv_data.chunks);
                LOG_DEBUG_MSG("parcel decode called for ZEROCOPY complete parcel");
            }
            else {
                throw std::runtime_error("RDMA Send completed with unmatched Id");
            }
        }

        // ----------------------------------------------------------------------------------------------
        // Every (signalled) rdma operation triggers a completion event when it completes,
        // the rdmaController calls this callback function and we must clean up all temporary
        // memory etc and signal hpx when sends or receives finish.
        // ----------------------------------------------------------------------------------------------
        int handle_verbs_completion(const struct ibv_wc completion, RdmaClient *client)
        {
            uint64_t wr_id = completion.wr_id;
            LOG_DEBUG_MSG("Handle verbs completion " << hexpointer(wr_id) << RdmaCompletionQueue::wc_opcode_str(completion.opcode));

            if (completion.opcode==IBV_WC_SEND) {
                LOG_DEBUG_MSG("Handle general completion " << hexpointer(wr_id) << RdmaCompletionQueue::wc_opcode_str(completion.opcode));
                handle_send_completion(wr_id);
                return 0;
            }

            //
            // When an Rdma Get operation completes, either add it to an ongoing parcel
            // receive, or if it is the last one, trigger decode message
            //
            else if (completion.opcode==IBV_WC_RDMA_READ) {
                handle_rdma_get_completion(wr_id, client);
                return 0;
            }

            //
            // A zero byte receive indicates we are being informed that remote GET operations are complete
            // we can release any data we were holding onto and signal a send as finished.
            // On hardware that do not support immediate data, a 4 byte tag message is used.
            //

            else if (completion.opcode==IBV_WC_RECV && completion.byte_len<=4) {
                handle_tag_recv_completion(wr_id, completion.imm_data, client);
                return 0;
            }

            //
            // When an unmatched receive completes, it is a new parcel, if everything fits into
            // the header, call decode message, otherwise, queue all the Rdma Get operations
            // necessary to complete the message
            //
            else if (completion.opcode==IBV_WC_RECV) {
                LOG_DEBUG_MSG("Entering receive (completion handler) section with received size " << decnumber(completion.byte_len));
                handle_recv_completion(wr_id, client);
                return 0;
            }

            throw std::runtime_error("Unexpected opcode in handle_verbs_completion ");
            //        << RdmaConnection::wr_opcode_str((ibv_wr_opcode)(completion.opcode)).c_str());
            return 0;
        }

        ~parcelport() {
            FUNC_START_DEBUG_MSG;
            _rdmaController = nullptr;
            FUNC_END_DEBUG_MSG;
        }

        /// Should not be used any more as parcelport_impl handles this
        bool can_bootstrap() const {
            FUNC_START_DEBUG_MSG;
            FUNC_END_DEBUG_MSG;
            return true;
        }

        /// Return the name of this locality
        std::string get_locality_name() const {
            FUNC_START_DEBUG_MSG;

            FUNC_END_DEBUG_MSG;
            // return ip address + ?
            return "verbs";
        }

        parcelset::locality agas_locality(util::runtime_configuration const & ini) const {
            FUNC_START_DEBUG_MSG;
            // load all components as described in the configuration information
            if (ini.has_section("hpx.agas")) {
                util::section const* sec = ini.get_section("hpx.agas");
                if (nullptr != sec) {
                    struct in_addr buf;
                    std::string addr = sec->get_entry("address", HPX_INITIAL_IP_ADDRESS);
                    LOG_DEBUG_MSG("Got AGAS addr " << addr.c_str();)
                    inet_pton(AF_INET, &addr[0], &buf);
                    return
                        parcelset::locality(locality(buf.s_addr));
                }
            }
            FUNC_END_DEBUG_MSG;
            LOG_DEBUG_MSG("Returning NULL agas locality")
            return parcelset::locality(locality(0xFFFF));
        }

        parcelset::locality create_locality() const {
            FUNC_START_DEBUG_MSG;
            FUNC_END_DEBUG_MSG;
            return parcelset::locality(locality());
        }
/*
        // should not be used any more.
        void put_parcels(std::vector<parcelset::locality> dests,
                std::vector<parcel> parcels,
                std::vector<write_handler_type> handlers)
        {
            FUNC_START_DEBUG_MSG;
            HPX_ASSERT(dests.size() == parcels.size());
            HPX_ASSERT(dests.size() == handlers.size());
            for(std::size_t i = 0; i != dests.size(); ++i)
            {
                put_parcel(dests[i], std::move(parcels[i]), handlers[i]);
            }
            FUNC_END_DEBUG_MSG;
        }
*/
        // This should start the receiving side of your PP
/*
        bool run(bool blocking = true) {
            FUNC_START_DEBUG_MSG;
            FUNC_END_DEBUG_MSG;
            return true;
        }
*/
        void do_stop() {
            LOG_DEBUG_MSG("Entering verbs stop ");
            FUNC_START_DEBUG_MSG;
            if (!stopped_) {
                bool finished = false;
                do {
                    finished = active_sends.empty() && active_recvs.empty();
                    if (!finished) {
                        LOG_ERROR_MSG("Entering STOP when not all parcels have completed");
                        std::terminate();
                    }
                } while (!finished && !hpx::is_stopped());

                unique_lock(stop_mutex);
                LOG_DEBUG_MSG("Removing all initiated connections");
                _rdmaController->removeAllInitiatedConnections();

                // wait for all clients initiated elsewhere to be disconnected
                while (_rdmaController->num_clients()!=0 && !hpx::is_stopped()) {
                    _rdmaController->eventMonitor(0);
                    std::cout << "Polling before shutdown" << std::endl;
                }
                LOG_DEBUG_MSG("stopped removing clients and terminating");
            }
            stopped_ = true;
            // Stop receiving and sending of parcels
        }

        // ----------------------------------------------------------------------------------------------
        // find the client queue pair object that is at the destination ip address
        // if no connection has been made yet, make one.
        // ----------------------------------------------------------------------------------------------
        RdmaClient *get_remote_connection(parcelset::locality const& dest)
        {
            boost::uint32_t dest_ip = dest.get<locality>().ip_;
            // @TODO, don't need smartpointers here, remove them as they waste an atomic refcount
            RdmaClientPtr client;
            {
                // lock this region as we are creating a connection to a remote locality
                // if two threads attempt to do this at the same time, we'll get duplicated clients
                unique_lock lock(connection_mutex);
                do {
                    ip_map_iterator ip_it = ip_qp_map.find(dest_ip);
                    if (ip_it!=ip_qp_map.end()) {
                        LOG_DEBUG_MSG("Connection found with qp " << ip_it->second);
                        client = _rdmaController->getClient(ip_it->second);
                        return client.get();
                    }
                    else {
                        if (connection_started.test_and_set(std::memory_order_acquire)) {
                            lock.unlock();
                            LOG_ERROR_MSG("A connection race has been detected, do not connect");
                            hpx::this_thread::sleep_for(std::chrono::milliseconds(1000));
                            lock.lock();
                        }
                        else {
                            LOG_DEBUG_MSG("Connection required to " << ipaddress(dest_ip));
                            client = _rdmaController->makeServerToServerConnection(dest_ip, _rdmaController->getPort());
                            LOG_DEBUG_MSG("Setting qpnum in main client map");
                            ip_qp_map[dest_ip] = client->getQpNum();
                            return client.get();
                        }
                    }
                } while (true);
            }
            return NULL;
        }
/*
        // ----------------------------------------------------------------------------------------------
        // called by hpx when an action is invoked on a remote locality.
        // This must be thread safe in order to function as any thread may invoke an action

        // not called any more : parcelport_impl?
        // ----------------------------------------------------------------------------------------------
        void put_parcel(parcelset::locality const & dest, parcel p, write_handler_type f) {
            FUNC_START_DEBUG_MSG;
            //
            // All the parcelport code should run on hpx threads and not OS threads
            //
            if (threads::get_self_ptr() == 0) {
                // this is an OS thread, so call put_parcel on an hpx thread
                hpx::apply(&parcelport::put_parcel, this, dest, std::move(p), f);
                return;
            }
            {   // if we already have a lot of sends in the queue, process them first
                while (active_send_count_ >= HPX_PARCELPORT_VERBS_MAX_SEND_QUEUE) {
//                    LOG_DEBUG_MSG("Extra HPX Background work");
                    hpx_background_work();
                }
            }

            {
                LOG_DEBUG_MSG("Extracting futures from parcel");
                std::shared_ptr<hpx::serialization::detail::future_await_container>
                    future_await(new hpx::serialization::detail::future_await_container());
                std::shared_ptr<hpx::serialization::output_archive>
                    archive(
                        new hpx::serialization::output_archive(
                            *future_await, 0, 0, 0, 0, &future_await->new_gids_)
                    );
                (*archive) << p;

                if(future_await->has_futures())
                {
                    void (parcelport::*awaiter)(
                      parcelset::locality const &, parcel, write_handler_type, bool
                      , std::shared_ptr<hpx::serialization::output_archive> const &
                      , std::shared_ptr<
                            hpx::serialization::detail::future_await_container> const &
                    )
                        = &parcelport::put_parcel_impl;
                    (*future_await)(
                        util::bind(
                            util::one_shot(awaiter), this,
                            dest, std::move(p), std::move(f), true,
                            archive, future_await)
                    );
                    return;
                }
                else
                {
                    LOG_DEBUG_MSG("About to send parcel");
                    put_parcel_impl(dest, std::move(p), std::move(f), true,
                      archive, future_await);
                }
            }
        }
*/
        template <typename Handler>
        bool async_write(Handler && handler, sender_connection *sender, snd_buffer_type &buffer)
        {
            FUNC_START_DEBUG_MSG;
            // if the serialization overflows the block, panic and rewrite this.
            {
                // create a tag, needs to be unique per client
                uint32_t tag = tag_provider_.next(sender->dest_ip_);
                LOG_DEBUG_MSG("Generated tag " << hexuint32(tag) << " from " << hexuint32(sender->dest_ip_));

                // we must store details about this parcel so that all memory buffers can be kept
                // until all send operations have completed.
                active_send_iterator current_send;
                {
                    unique_lock lock(active_send_mutex);
                    // if more than N parcels are currently queued, then yield
                    // otherwise we can fill the send queues with so many requests
                    // that memory buffers are exhausted.
                    LOG_DEBUG_MSG("HPX_PARCELPORT_VERBS_MAX_SEND_QUEUE " << HPX_PARCELPORT_VERBS_MAX_SEND_QUEUE << " sends " << hexnumber(active_sends.size()));
                    active_send_condition.wait(lock, [this] {
                      return (active_sends.size()<HPX_PARCELPORT_VERBS_MAX_SEND_QUEUE);
                    });

                    active_sends.emplace_back();
                    current_send = std::prev(active_sends.end());
                    ++active_send_count_;
                    LOG_DEBUG_MSG("Active send after insert size " << hexnumber(active_send_count_));
                }
                parcel_send_data &send_data = *current_send;
                send_data.tag            = tag;
                send_data.handler        = std::move(handler);
                send_data.header_region  = NULL;
                send_data.message_region = NULL;
                send_data.chunk_region   = NULL;
                send_data.has_zero_copy  = false;
                send_data.delete_flag.clear();

                LOG_DEBUG_MSG("Generated unique dest " << hexnumber(sender->dest_ip_) << " coded tag " << hexuint32(send_data.tag));

                // for each zerocopy chunk, we must create a memory region for the data
                for (serialization::serialization_chunk &c : buffer.chunks_) {
                    LOG_DEBUG_MSG("write : chunk : size " << hexnumber(c.size_)
                            << " type " << decnumber((uint64_t)c.type_)
                            << " rkey " << decnumber(c.rkey_)
                            << " cpos " << hexpointer(c.data_.cpos_)
                            << " pos " << hexpointer(c.data_.pos_)
                            << " index " << decnumber(c.data_.index_));
                }

                // for each zerocopy chunk, we must create a memory region for the data
                int index = 0;
                for (serialization::serialization_chunk &c : buffer.chunks_) {
                    if (c.type_ == serialization::chunk_type_pointer) {
                        send_data.has_zero_copy  = true;
                        // if the data chunk fits into a memory block, copy it
                        util::high_resolution_timer regtimer;
                        RdmaMemoryRegion *zero_copy_region;
                        if (c.size_<=HPX_PARCELPORT_VERBS_MEMORY_COPY_THRESHOLD) {
                            zero_copy_region = chunk_pool_->allocateRegion(std::max(c.size_, (std::size_t)RDMA_DEFAULT_MEMORY_POOL_SMALL_CHUNK_SIZE));
                            char *zero_copy_memory = (char*)(zero_copy_region->getAddress());
                            std::memcpy(zero_copy_memory, c.data_.cpos_, c.size_);
                            // the pointer in the chunk info must be changed
                            buffer.chunks_[index] = serialization::create_pointer_chunk(zero_copy_memory, c.size_);
//                            LOG_DEBUG_MSG("Time to copy memory (ns) " << decnumber(regtimer.elapsed_nanoseconds()));
                        }
                        else {
                            // create a memory region from the pointer
                            zero_copy_region = new RdmaMemoryRegion(
                                    _rdmaController->getProtectionDomain(), c.data_.cpos_, std::max(c.size_, (std::size_t)RDMA_DEFAULT_MEMORY_POOL_SMALL_CHUNK_SIZE));
//                            LOG_DEBUG_MSG("Time to register memory (ns) " << decnumber(regtimer.elapsed_nanoseconds()));
                        }
                        c.rkey_  = zero_copy_region->getRemoteKey();
                        LOG_DEBUG_MSG("Zero-copy rdma Get region " << decnumber(index) << " created for address "
                                << hexpointer(zero_copy_region->getAddress())
                                << " and rkey " << decnumber(c.rkey_));
                        send_data.zero_copy_regions.push_back(zero_copy_region);
                    }
                    index++;
                }

                // grab a memory block from the pinned pool to use for the header
                send_data.header_region = chunk_pool_->allocateRegion(chunk_pool_->small_.chunk_size_);
                char *header_memory = (char*)(send_data.header_region->getAddress());

                // create the header in the pinned memory block
                LOG_DEBUG_MSG("Placement new for the header with piggyback copy disabled");
                header_type *h = new(header_memory) header_type(buffer, send_data.tag);
                h->assert_valid();
                send_data.header_region->setMessageLength(h->header_length());

                LOG_DEBUG_MSG(
                        "sending, buffsize " << decnumber(h->size())
                        << "header_length " << decnumber(h->header_length())
                        << "numbytes " << decnumber(h->numbytes())
                        << "chunks zerocopy( " << decnumber(h->num_chunks().first) << ") "
                        << ", chunk_flag " << decnumber(h->header_length())
                        << ", normal( " << decnumber(h->num_chunks().second) << ") "
                        << ", chunk_flag " << decnumber(h->header_length())
                        << "tag " << hexuint32(h->tag())
                );

                // Get the block of pinned memory where the message was encoded during serialization
                send_data.message_region = buffer.data_.m_region_;
                send_data.message_region->setMessageLength(h->size());
                LOG_DEBUG_MSG("Found region allocated during encode_parcel : address " << hexpointer(buffer.data_.m_array_) << " region "<< hexpointer(send_data.message_region));

                // header region is always sent, message region is usually piggybacked
                int num_regions = 1;
                RdmaMemoryRegion *region_list[2] = { send_data.header_region, send_data.message_region };
                if (h->chunk_data()) {
                    LOG_DEBUG_MSG("Chunk info is piggybacked");
                    send_data.chunk_region = NULL;
                }
                else {
                    throw std::runtime_error(
                            "@TODO : implement chunk info rdma get when zero-copy chunks exceed header space");
                }

                if (h->piggy_back()) {
                    LOG_DEBUG_MSG("Main message is piggybacked");
                    num_regions += 1;
                }
                else {
                    LOG_DEBUG_MSG("Main message NOT piggybacked ");
                    h->setRdmaMessageLength(h->size());
                    h->setRdmaKey(send_data.message_region->getLocalKey());
                    h->setRdmaAddr(send_data.message_region->getAddress());
                    send_data.zero_copy_regions.push_back(send_data.message_region);
                    send_data.has_zero_copy  = true;
                    LOG_DEBUG_MSG("RDMA message " << hexnumber(buffer.data_.size())
                            << " rkey " << decnumber(send_data.message_region->getLocalKey())
                            << " pos " << hexpointer(send_data.message_region->getAddress()));
                    // do not delete twice, clear the message block pointer as it
                    // is also used in the zero_copy_regions list
                    send_data.message_region = NULL;
                }

                uint64_t wr_id = (uint64_t)(send_data.header_region);
                {
                    // add wr_id's to completion map
                    unique_lock lock(SendCompletionMap_mutex);
#ifdef HPX_DEBUG
                    if (SendCompletionMap.find(wr_id) != SendCompletionMap.end()) {
                        for (auto & pair : SendCompletionMap) {
                            std::cout << hexpointer(pair.first) << "\n";
                        }
                        LOG_ERROR_MSG("FATAL : wr_id duplicated " << hexpointer(wr_id));
                        std::terminate();
                        throw std::runtime_error("wr_id duplicated in put_parcel : FATAL");
                    }
#endif
                    // put everything into map to be retrieved when send completes
                    SendCompletionMap[wr_id] = current_send;
                    LOG_DEBUG_MSG("wr_id added to SendCompletionMap "
                            << hexpointer(wr_id) << " Entries " << SendCompletionMap.size());
                }
                {
                    // if there are zero copy regions (or message/chunks are not piggybacked),
                    // we must hold onto the regions until the destination tells us
                    // it has completed all rdma Get operations
                    if (send_data.has_zero_copy) {
                        scoped_lock lock(TagSendCompletionMap_mutex);
                        // put the data into a new map which is indexed by the Tag of the send
                        // zero copy blocks will be released when we are told this has completed
                        TagSendCompletionMap[send_data.tag] = current_send;
                    }
                }

                // send the header/main_chunk to the destination, wr_id is header_region (entry 0 in region_list)
                LOG_TRACE_MSG("num regions to send " << num_regions
                    << "Block header_region"
                    << " buffer "    << hexpointer(send_data.header_region->getAddress())
                    << " region "    << hexpointer(send_data.header_region));
                if (num_regions>1) {
                    LOG_TRACE_MSG(
                    "Block message_region "
                    << " buffer "    << hexpointer(send_data.message_region->getAddress())
                    << " region "    << hexpointer(send_data.message_region));
                }
                sender->client_->postSend_xN(region_list, num_regions, true, false, 0);

                // log the time spent in performance counter
//                buffer.data_point_.time_ =
//                        timer.elapsed_nanoseconds() - buffer.data_point_.time_;

                // parcels_sent_.add_data(buffer.data_point_);
            }
            FUNC_END_DEBUG_MSG;
            return true;
        }
/*
        void put_parcel_impl(
            parcelset::locality const & dest, parcel p, write_handler_type f, bool trigger
          , std::shared_ptr<hpx::serialization::output_archive> const & archive
          , std::shared_ptr<
                hpx::serialization::detail::future_await_container
            > const & future_await)
        {
            boost::uint32_t dest_ip = dest.get<locality>().ip_;
            LOG_DEBUG_MSG("Locality " << ipaddress(_ibv_ip) << " put_parcel_impl to " << ipaddress(dest_ip) );

            FUNC_END_DEBUG_MSG;
        }
*/

        // ----------------------------------------------------------------------------------------------
        // This is called to poll for completions and handle all incoming messages as well as complete
        // outgoing messages.
        //
        // @TODO : It is assumed that hpx_bakground_work is only going to be called from
        // an hpx thread.
        // Since the parcelport can be serviced by hpx threads or by OS threads, we must use extra
        // care when dealing with mutexes and condition_variables since we do not want to suspend an
        // OS thread, but we do want to suspend hpx threads when necessary.
        // ----------------------------------------------------------------------------------------------
        bool hpx_background_work() {
            bool done = false;
            // this must be called on an HPX thread
//            HPX_ASSERT(threads::get_self_ptr() != 0);
            //
            do {
                // if an event comes in, we may spend time processing/handling it and another may arrive
                // during this handling, so keep checking until none are received
                done = (_rdmaController->eventMonitor(0) == 0);
            } while (!done);
            return true;
        }

        // --------------------------------------------------------------------
        // Background work, HPX thread version, used on custom scheduler
        // to poll in an OS background thread which is pre-emtive and therefore
        // could help reduce latencies (when hpx threads are all doing work).
        // --------------------------------------------------------------------
#ifdef USE_SPECIALIZED_SCHEDULER
        void hpx_background_work_thread()
        {
            // repeat until no more parcels are to be sent
            while (!stopped_ || !hpx::is_stopped()) {
                hpx_background_work();
            }
            LOG_DEBUG_MSG("hpx background work thread stopped");
        }
#endif

        // --------------------------------------------------------------------
        // Background work, OS thread version
        // --------------------------------------------------------------------
        static inline bool OS_background_work() {
            return parcelport::get_singleton()->hpx_background_work();
        }

        // ----------------------------------------------------------------------------------------------
        // This is called whenever a HPX OS thread is idling, can be used to poll for incoming messages
        // this should be thread safe as eventMonitor only polls and dispatches and is thread safe.
        // ----------------------------------------------------------------------------------------------
        bool background_work(std::size_t num_thread) {
            if (stopped_ || hpx::is_stopped()) {
                return false;
			            }
            return hpx::apply(&parcelport::OS_background_work);
        }

        static parcelport *get_singleton()
        {
            return _parcelport_instance;
        }

        static parcelport *_parcelport_instance;

    };


    template <typename Handler, typename ParcelPostprocess>
    void sender_connection::async_write(Handler && handler, ParcelPostprocess && parcel_postprocess)
    {
        HPX_ASSERT(!buffer_.data_.empty());
        //
        postprocess_handler_ = std::forward<ParcelPostprocess>(parcel_postprocess);
        //
        if (!parcelport_->async_write(std::move(handler), this, buffer_)) {
            // after send has done, setup a fresh buffer for next time
            LOG_DEBUG_MSG("Wiping buffer 1");

            snd_data_type pinned_vector(chunk_pool_);
            snd_buffer_type buffer(std::move(pinned_vector), chunk_pool_);
            buffer_ = std::move(buffer);
            error_code ec;
            postprocess_handler_(ec, there_, shared_from_this());
        }
        else {
            // after send has done, setup a fresh buffer for next time
            LOG_DEBUG_MSG("Wiping buffer 2");
            snd_data_type pinned_vector(chunk_pool_);
            snd_buffer_type buffer(std::move(pinned_vector), chunk_pool_);
            buffer_ = std::move(buffer);
            error_code ec;
            postprocess_handler_(ec, there_, shared_from_this());
        }
    }

}}}}

namespace hpx {
namespace traits {
// Inject additional configuration data into the factory registry for this
// type. This information ends up in the system wide configuration database
// under the plugin specific section:
//
//      [hpx.parcel.verbs]
//      ...
//      priority = 100
//
template<>
struct plugin_config_data<hpx::parcelset::policies::verbs::parcelport> {
    static char const* priority() {
        FUNC_START_DEBUG_MSG;
        static int log_init = false;
        if (!log_init) {
#ifdef RDMAHELPER_HAVE_LOGGING
            initRdmaHelperLogging();
#endif
            log_init = true;
        }
        FUNC_END_DEBUG_MSG;
        return "100";
    }

    // This is used to initialize your parcelport, for example check for availability of devices etc.
    static void init(int *argc, char ***argv, util::command_line_handling &cfg) {
        FUNC_START_DEBUG_MSG;

        FUNC_END_DEBUG_MSG;
    }

    static char const* call() {
        FUNC_START_DEBUG_MSG;
        FUNC_END_DEBUG_MSG;

        //      LOG_DEBUG_MSG("\n"
        //          "ifname = ${HPX_PARCELPORT_VERBS_IFNAME:" HPX_PARCELPORT_VERBS_IFNAME "}\n"
        //          "device = ${HPX_PARCELPORT_VERBS_DEVICE:" HPX_PARCELPORT_VERBS_DEVICE "}\n"
        //          "interface = ${HPX_PARCELPORT_VERBS_INTERFACE:" HPX_PARCELPORT_VERBS_INTERFACE "}\n"
        //          "memory_chunk_size = ${HPX_PARCEL_IBVERBS_MEMORY_CHUNK_SIZE:"
        //          BOOST_PP_STRINGIZE(HPX_PARCELPORT_VERBS_MEMORY_CHUNK_SIZE) "}\n"
        //          "max_memory_chunks = ${HPX_PARCEL_IBVERBS_MAX_MEMORY_CHUNKS:"
        //          BOOST_PP_STRINGIZE(HPX_PARCELPORT_VERBS_MAX_MEMORY_CHUNKS) "}\n"
        //          "zero_copy_optimization = 0\n"
        //          "io_pool_size = 2\n"
        //          "use_io_pool = 1\n"
        //          "enable = 0");
        return
//                "ifname = ${HPX_PARCELPORT_VERBS_IFNAME:" HPX_PARCELPORT_VERBS_IFNAME "}\n"
                "device = ${HPX_PARCELPORT_VERBS_DEVICE:" HPX_PARCELPORT_VERBS_DEVICE "}\n"
                "interface = ${HPX_PARCELPORT_VERBS_INTERFACE:" HPX_PARCELPORT_VERBS_INTERFACE "}\n"
                "memory_chunk_size = ${HPX_PARCELPORT_VERBS_MEMORY_CHUNK_SIZE:"
                BOOST_PP_STRINGIZE(HPX_PARCELPORT_VERBS_MEMORY_CHUNK_SIZE) "}\n"
        "max_memory_chunks = ${HPX_PARCELPORT_VERBS_MAX_MEMORY_CHUNKS:"
        BOOST_PP_STRINGIZE(HPX_PARCELPORT_VERBS_MAX_MEMORY_CHUNKS) "}\n"
        "zero_copy_optimization = 1\n"
        "io_pool_size = 2\n"
        "use_io_pool = 1\n"
        "enable = 0"
        ;
    }
};
}
}
RdmaControllerPtr hpx::parcelset::policies::verbs::parcelport::_rdmaController;
//std::string       hpx::parcelset::policies::verbs::parcelport::_ibverbs_ifname;
std::string       hpx::parcelset::policies::verbs::parcelport::_ibverbs_device;
std::string       hpx::parcelset::policies::verbs::parcelport::_ibverbs_interface;
boost::uint32_t   hpx::parcelset::policies::verbs::parcelport::_ibv_ip;
boost::uint32_t   hpx::parcelset::policies::verbs::parcelport::_port;
hpx::parcelset::policies::verbs::parcelport *hpx::parcelset::policies::verbs::parcelport::_parcelport_instance;

HPX_REGISTER_PARCELPORT(hpx::parcelset::policies::verbs::parcelport, verbs);
