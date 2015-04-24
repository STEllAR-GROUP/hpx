//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Thomas Heller
//  Copyright (c) 2015 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// config
#include <hpx/config/defines.hpp>
#include <hpx/hpx_fwd.hpp>

// util
#include <hpx/util/command_line_handling.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/util/safe_bool.hpp>
#include <hpx/util/memory_chunk_pool_allocator.hpp>

// The memory pool specialization need to be pulled in before encode_parcels
#include "RdmaMemoryPool.h"

// parcelport
#include <hpx/runtime.hpp>
#include <hpx/runtime/parcelset/parcelport.hpp>
#include <hpx/runtime/parcelset/parcel_buffer.hpp>
#include <hpx/runtime/parcelset/encode_parcels.hpp>
#include <hpx/runtime/parcelset/decode_parcels.hpp>
#include <hpx/plugins/parcelport_factory.hpp>

// Local parcelport plugin
#include "connection_handler_verbs.hpp"
#include "pointer_wrapper_vector.hpp"
//
#include <hpx/plugins/parcelport/header.hpp>

// rdmahelper library
#include "RdmaLogging.h"
#include "RdmaController.h"
#include "RdmaDevice.h"
//
#include <unordered_map>
//
#include <mutex>
#include <condition_variable>

#define HPX_PARCELPORT_VERBS_MEMORY_COPY_THRESHOLD DEFAULT_MEMORY_POOL_CHUNK_SIZE
//#define HPX_PARCELPORT_VERBS_MEMORY_COPY_THRESHOLD 256

using namespace hpx::parcelset::policies;

namespace hpx { namespace parcelset { namespace policies { namespace verbs
{
    // ----------------------------------------------------------------------------------------------
    // Locality, represented by an ip address and a queue pair (qp) id
    // the qp id is used internally for quick lookup to find the peer
    // that we want to communicate with
    // ----------------------------------------------------------------------------------------------
    struct locality {
      static const char *type() {
        return "verbs";
      }

      explicit locality(boost::uint32_t ip, boost::uint32_t port) :
            ip_(ip), port_(port), qp_(0xFFFF) {}

      locality() : ip_(0xFFFF), port_(0), qp_(0xFFFF) {}

      // some condition marking this locality as valid
      operator util::safe_bool<locality>::result_type() const {
        return util::safe_bool<locality>()(ip_ != boost::uint32_t(0xFFFF));
      }

      void save(serialization::output_archive & ar) const {
        ar.save(ip_);
      }

      void load(serialization::input_archive & ar) {
        ar.load(ip_);
      }

    private:
      friend bool operator==(locality const & lhs, locality const & rhs) {
        return lhs.ip_ == rhs.ip_;
      }

      friend bool operator<(locality const & lhs, locality const & rhs) {
        return lhs.ip_ < rhs.ip_;
      }

      friend std::ostream & operator<<(std::ostream & os, locality const & loc) {
        boost::io::ios_flags_saver
        ifs(os);
        os << loc.ip_;
        return os;
      }
    public:
      boost::uint32_t ip_;
      boost::uint32_t port_;
      boost::uint32_t qp_;
    };

    // ----------------------------------------------------------------------------------------------
    // simple atomic counter we use for tags
    // when a parcel is sent to a remote locality, it may need to pull zero copy chunks from us.
    // we keep the chunks until the remote locality sends a zero byte message with the tag we gave
    // them and then we know it is safe to release the memory back to the pool.
    // The tags can have a short lifetime, but must be unique, so we encode the ip address with
    // a counter to generate tags per destination.
    // The tag is send in immediate data so must be 32bits only : Note that the tag only has a
    // lifetime of the unprocessed parcel, so it can be reused as soon as the parcel has been completed
    // and herefore a 16bit count is sufficient as we only keep a few parcels per locality in flight at a time
    // ----------------------------------------------------------------------------------------------
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

    // ----------------------------------------------------------------------------------------------
    // parcelport, the implementation of the parcelport itself
    // ----------------------------------------------------------------------------------------------
    class parcelport: public parcelset::parcelport {
    private:

        // ----------------------------------------------------------------------------------------------
        // returns a locality object that represents 'this' locality
        // ----------------------------------------------------------------------------------------------
        static parcelset::locality here(util::runtime_configuration const& ini) {
            FUNC_START_DEBUG_MSG;
            if (ini.has_section("hpx.parcel.verbs")) {
                util::section const * sec = ini.get_section("hpx.parcel.verbs");
                if (NULL != sec) {
                    std::string ibverbs_enabled(sec->get_entry("enable", "0"));
                    if (boost::lexical_cast<int>(ibverbs_enabled)) {
                        _ibverbs_ifname    = sec->get_entry("ifname",    HPX_PARCELPORT_VERBS_IFNAME);
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
                    LOG_DEBUG_MSG("hpx.agas port number " << hpx::util::get_entry_as<boost::uint16_t>(*sec, "port", HPX_INITIAL_IP_PORT));
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
            return parcelset::locality(locality(_ibv_ip, _ibv_ip));
        }

    public:
        // ----------------------------------------------------------------------------------------------
        // Constructor : mostly just initializes the superclass with 'here'
        // ----------------------------------------------------------------------------------------------
        parcelport(util::runtime_configuration const& ini,
                util::function_nonser<void(std::size_t, char const*)> const& on_start_thread,
                util::function_nonser<void()> const& on_stop_thread) :
                    parcelset::parcelport(ini, here(ini), "verbs"), archive_flags_(0)
        , stopped_(false)
        //      , parcels_sent_(0)
        {
            FUNC_START_DEBUG_MSG;
            //    _port   = 0;
#ifdef BOOST_BIG_ENDIAN
            std::string endian_out = get_config_entry("hpx.parcel.endian_out", "big");
#else
            std::string endian_out = get_config_entry("hpx.parcel.endian_out", "little");
#endif
            if (endian_out == "little")
                archive_flags_ |= serialization::endian_little;
            else if (endian_out == "big")
                archive_flags_ |= serialization::endian_big;
            else {
                HPX_ASSERT(endian_out == "little" || endian_out == "big");
            }

            if (!this->allow_array_optimizations()) {
                archive_flags_ |= serialization::disable_array_optimization;
                archive_flags_ |= serialization::disable_data_chunking;
                LOG_DEBUG_MSG("Disabling array optimization and data chunking");
            } else {
                if (!this->allow_zero_copy_optimizations()) {
                    archive_flags_ |= serialization::disable_data_chunking;
                    LOG_DEBUG_MSG("Disabling data chunking");
                }
            }
            _rdmaController = std::make_shared<RdmaController>
                (_ibverbs_device.c_str(), _ibverbs_interface.c_str(), _port);

            FUNC_END_DEBUG_MSG;
        }

        static RdmaControllerPtr _rdmaController;
        static std::string       _ibverbs_ifname;
        static std::string       _ibverbs_device;
        static std::string       _ibverbs_interface;
        static boost::uint32_t   _port;
        static boost::uint32_t   _ibv_ip;
        //
        typedef std::map<boost::uint32_t, boost::uint32_t> ip_map;
        typedef ip_map::iterator                           ip_map_iterator;
        //
        ip_map ip_qp_map;

        // @TODO, clean up the allocators, buffers, chunk_pool etc so that there is a more consistent
        // reuse of classes/types
        typedef header<DEFAULT_MEMORY_POOL_CHUNK_SIZE>  header_type;
        typedef hpx::lcos::local::spinlock              mutex_type;
        typedef hpx::lcos::local::spinlock::scoped_lock lock_type;
        typedef hpx::lcos::local::condition_variable    condition_type;

        // use std::mutex in stop function as HPX is terminating
        std::mutex                  stop_mutex;
        std::condition_variable     stop_cond;
        hpx::lcos::local::spinlock  connection_mutex;
        hpx::lcos::local::spinlock  TagReceiveCompletionMap_mutex;
        hpx::lcos::local::spinlock  ZeroCopyGetCompletionMap_mutex;
        hpx::lcos::local::spinlock  SendCompletionMap_mutex;
        hpx::lcos::local::spinlock  TagSendCompletionMap_mutex;

        typedef char                                                      memory_type;
        typedef RdmaMemoryPool                                            memory_pool_type;
        typedef std::shared_ptr<memory_pool_type>                         memory_pool_ptr_type;
        typedef hpx::util::detail::memory_chunk_pool_allocator
                <memory_type, memory_pool_type, mutex_type>               allocator_type;
        typedef std::vector<memory_type, allocator_type>                  snd_data_type;
        typedef util::detail::pointer_wrapper_vector<memory_type>         rcv_data_type;
        typedef parcel_buffer<snd_data_type>                              snd_buffer_type;
        typedef parcel_buffer<rcv_data_type, std::vector<memory_type>>    rcv_buffer_type;

        //
        int                       archive_flags_;
        boost::atomic<bool>       stopped_;
        memory_pool_ptr_type      chunk_pool_;
        verbs::tag_provider       tag_provider_;

        // performance_counters::parcels::gatherer& parcels_sent_;


        // ----------------------------------------------------------------------------------------------
        // each region allocated during a parcel receive will need to be released when the receive
        // has completed : recombining of final parcel depends on which piece of the message the
        // regions belong to
        // ----------------------------------------------------------------------------------------------
        typedef struct {
            int               counter;
            std::vector<serialization::serialization_chunk>  chunks;
            RdmaMemoryRegion *header_region, *chunk_region, *message_region;
            std::vector<RdmaMemoryRegion*>                   zero_copy_regions;
        } parcel_receive_data;

        // ----------------------------------------------------------------------------------------------
        // each region allocated during a parcel send will need to be released when the send has completed
        // ----------------------------------------------------------------------------------------------
        typedef struct {
            uint32_t                                      tag;
            parcelset::parcel                             parcel;
            parcelset::parcelhandler::write_handler_type  handler;
            RdmaMemoryRegion *header_region, *chunk_region, *message_region;
            std::vector<RdmaMemoryRegion*>                   zero_copy_regions;
        } parcel_send_data;

        // map wr_id to receive message tag
        typedef std::unordered_map<uint64_t, uint32_t> rdmaget_wr_tag_map;
        // map receive message tag to the receive data
        typedef std::unordered_map<uint32_t, parcel_receive_data> tag_receive_map;

        // map rdma get to tag
        rdmaget_wr_tag_map ZeroCopyGetCompletionMap;
        // map tag to receive struct
        tag_receive_map    TagReceiveCompletionMap;

        typedef std::list<parcel_send_data> active_list_type;
        typedef active_list_type::iterator  active_send_iterator;

        // map send parcel wr_id to all info needed on completion
        typedef std::unordered_map<uint64_t, active_send_iterator> send_wr_map;

        // store received objects using a map referenced by verbs work request ID
        send_wr_map SendCompletionMap;
        send_wr_map TagSendCompletionMap;

        active_list_type active_sends;
        mutex_type       active_send_mutex;

        // ----------------------------------------------------------------------------------------------
        // When a parcel_buffer finishes decoding a message, it will call this function
        // and we are free to release any memory buffers we were holding.
        // ----------------------------------------------------------------------------------------------
        void release_memory_buffers(uint32_t tag)
        {
            FUNC_START_DEBUG_MSG;
            hpx::lcos::local::spinlock::scoped_lock lk(TagReceiveCompletionMap_mutex);
            LOG_DEBUG_MSG("Received a request to release buffers with tag " << hexuint32(tag));
            tag_receive_map::iterator it_zt = TagReceiveCompletionMap.find(tag);
            if (it_zt==TagReceiveCompletionMap.end()) {
                LOG_DEBUG_MSG("RDMA Get tag " << hexuint32(tag) << " was not found during release_memory_buffers");
                std::terminate();
                throw std::runtime_error("RDMA GET tag not found in release_memory_buffers");
            }
            parcel_receive_data &r_data = it_zt->second;
            LOG_DEBUG_MSG("RDMA Get tag " << hexuint32(tag) << " has count of " << r_data.counter);

            chunk_pool_->deallocate(r_data.header_region);
            LOG_DEBUG_MSG("Zero copy regions size is (delete) " << r_data.zero_copy_regions.size());
            for (auto r : r_data.zero_copy_regions) {
                // if this region was registered on the fly, then don't return it to the pool
                if (r->isTempRegion()) {
                    LOG_DEBUG_MSG("Deleting " << hexpointer(r));
                    delete r;
                }
                else if (r->isUserRegion()) {
                    LOG_DEBUG_MSG("Deleting " << hexpointer(r));
                    delete r;
                }
                else {
                    LOG_DEBUG_MSG("Deallocating " << hexpointer(r));
                    chunk_pool_->deallocate(r);
                }
            }
            TagReceiveCompletionMap.erase(it_zt);
            LOG_DEBUG_MSG("release_memory_buffers done ");
            FUNC_END_DEBUG_MSG;
        }

        // ----------------------------------------------------------------------------------------------
        // clean up a completed send and all its regions etc
        // ----------------------------------------------------------------------------------------------
        void delete_send_data(active_send_iterator send) {
            parcel_send_data &send_data = *send;
            // trigger the write_handler
            LOG_DEBUG_MSG("Calling write_handler for completed send");
            send_data.handler.operator()(error_code(), send_data.parcel);
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
                // if this region was registered on the fly, then don't return it to the pool
                if (r->isTempRegion()) {
                    LOG_DEBUG_MSG("Deleting " << hexpointer(r));
                    delete r;
                }
                else if (r->isUserRegion()) {
                    LOG_DEBUG_MSG("Deleting " << hexpointer(r));
                    delete r;
                }
                else {
                    LOG_DEBUG_MSG("Deallocating " << hexpointer(r));
                    chunk_pool_->deallocate(r);
                }
            }
            // and trigger the send complete handler for hpx internal cleanup
            LOG_DEBUG_MSG("Calling write_handler for completed send");
            send_data.handler.operator()(error_code(), send_data.parcel);
            {
                hpx::lcos::local::spinlock::scoped_lock(active_send_mutex);
                LOG_DEBUG_MSG("erasing current send from active send list");
                active_sends.erase(send);
            }
        }

        // ----------------------------------------------------------------------------------------------
        // handler for connections, this is triggered as a callback from the rdmaController when
        // a connection event has occurred.
        // When we connect to another locality, all internal structures are updated accordingly,
        // but when another locality connects to us, we must do it manually via this callback
        // ----------------------------------------------------------------------------------------------
        int handle_verbs_connection(std::pair<uint32_t,uint64_t> qpinfo, RdmaClientPtr client)
        {
            boost::lock_guard<hpx::lcos::local::spinlock> lock(connection_mutex);
            LOG_DEBUG_MSG("handle_verbs_connection callback triggered");
            boost::uint32_t dest_ip = client->getRemoteIPv4Address();
            ip_map_iterator ip_it = ip_qp_map.find(dest_ip);
            if (ip_it==ip_qp_map.end()) {
                ip_qp_map.insert(std::make_pair(dest_ip, qpinfo.first));
                client->setInitiatedConnection(false);
                LOG_DEBUG_MSG("handle_verbs_connection OK adding " << ipaddress(dest_ip));
            }
            else {
                throw std::runtime_error("verbs parcelport : should not be receiving a connection more than once");
            }
            return 0;
        }

        // ----------------------------------------------------------------------------------------------
        // Every (signalled) rdma operation triggers a completion event when it completes,
        // the rdmaController calls this callback function and we must clean up all temporary
        // memory etc and signal hpx when sends or receives finish.
        // ----------------------------------------------------------------------------------------------
        int handle_verbs_completion(struct ibv_wc *completion, RdmaClientPtr client)
        {
            uint64_t wr_id = completion->wr_id;
            //
            send_wr_map::iterator         it_s;
            rdmaget_wr_tag_map::iterator  it_z;

            //
            // When a send completes, release memory and trigger write_handler
            //
            bool                 found_wr_id;
            RdmaMemoryRegion    *region;
            active_send_iterator current_send;
            {
                // make sure map isn't modified whilst we are querying it
                hpx::lcos::local::spinlock::scoped_lock lk(SendCompletionMap_mutex);
                it_s = SendCompletionMap.find(wr_id);
                found_wr_id = (it_s != SendCompletionMap.end());
                if (found_wr_id) {
                    LOG_DEBUG_MSG("Entering send (completion handler) section");
                    region = (RdmaMemoryRegion *)completion->wr_id;
                    current_send = it_s->second;
                    // after the send has completed, we can release memory regions used
                    // note : release iterator before memory regions to avoid race as region is used as map key
                    // and it might be re-added to the map with new data
                    LOG_DEBUG_MSG("erasing iterator for completed send");
                    SendCompletionMap.erase(it_s);
                }
            }
            if (found_wr_id) {
                // if the send had no zero_copy regions, then it has completed
                if (current_send->zero_copy_regions.empty()) {
                    delete_send_data(current_send);
                }
                return 0;
            }
            //
            // When an Rdma Get operation completes, either add it to an ongoing parcel
            // receive, or if it is the last one, trigger decode message
            //
            {
                // lock map access during find
                hpx::lcos::local::spinlock::scoped_lock lk(ZeroCopyGetCompletionMap_mutex);
                it_z = ZeroCopyGetCompletionMap.find(wr_id);
                found_wr_id = (it_z != ZeroCopyGetCompletionMap.end());
            }
            if (found_wr_id) {
                tag_receive_map::iterator it_zt;
                uint32_t tag = it_z->second;
                {   // lock section
                    hpx::lcos::local::spinlock::scoped_lock lk(TagReceiveCompletionMap_mutex);
                    it_zt = TagReceiveCompletionMap.find(tag);
                    if (it_zt==TagReceiveCompletionMap.end()) {
                        LOG_DEBUG_MSG("RDMA Get tag " << hexuint32(tag) << " was not found in completion tag map");
                        std::terminate();
                        throw std::runtime_error("RDMA GET tag not found : FATAL");
                    }
                    LOG_DEBUG_MSG("RDMA Get tag " << hexuint32(tag) << " has count of " << it_zt->second.counter);
                    if (--it_zt->second.counter > 0) {
                        // we can't do anything until all zero copy chunks are here
                        return 0;
                    }
                    // all Get operations have completed, so we can process the parcel
                    {
                        hpx::lcos::local::spinlock::scoped_lock lk2(ZeroCopyGetCompletionMap_mutex);
                        ZeroCopyGetCompletionMap.erase(it_z);
                    }
                }
                parcel_receive_data &r_data = it_zt->second;
                //
                LOG_DEBUG_MSG("RDMA Get tag " << hexuint32(tag) << " has completed : posting zero byte ack to origin");
                client->postSend_x0((RdmaMemoryRegion*)tag, false, true, tag);
                //
                LOG_DEBUG_MSG("Zero copy regions size is (completion) " << r_data.zero_copy_regions.size());

                header_type *h = (header_type*)r_data.header_region->getAddress();
                LOG_DEBUG_MSG( "get completion " <<
                        "buffsize " << decnumber(h->size())
                        << "numbytes " << decnumber(h->numbytes())
                        << "chunks zerocopy( " << decnumber(h->num_chunks().first) << ") "
                        << ", normal( " << decnumber(h->num_chunks().second) << ") "
                        << " chunkdata " << decnumber((h->chunk_data()!=NULL))
                        << " piggyback " << decnumber((h->piggy_back()!=NULL))
                        << " tag " << hexuint32(h->tag())
                );

                char *piggy_back = h->piggy_back();
                LOG_DEBUG_MSG("Creating a release buffer callback for tag " << hexuint32(tag));
                rcv_data_type wrapped_pointer(piggy_back, h->size(),
                        boost::bind(&parcelport::release_memory_buffers, this, tag));
                rcv_buffer_type buffer(std::move(wrapped_pointer));
                LOG_DEBUG_MSG("calling parcel decode for complete ZEROCOPY parcel");

                for (serialization::serialization_chunk &c : r_data.chunks) {
                    LOG_DEBUG_MSG("chunk : size " << hexnumber(c.size_)
                            << " type " << decnumber((uint64_t)c.type_)
                            << " rkey " << decnumber(c.rkey_)
                            << " cpos " << hexpointer(c.data_.cpos_)
                            << " pos " << hexpointer(c.data_.pos_)
                            << " index " << decnumber(c.data_.index_));
                }

                buffer.num_chunks_ = h->num_chunks();
                buffer.data_.resize(static_cast<std::size_t>(h->size()));
                buffer.data_size_ = h->size();
                buffer.chunks_.resize(r_data.chunks.size());
                decode_message_with_chunks(*this, std::move(buffer), 1, r_data.chunks);
                LOG_DEBUG_MSG("parcel decode called for ZEROCOPY complete parcel");
            }
            //
            // a zero byte receive indicates we are being informed that remote GET operations are complete
            // we can release any data we were holding onto and signal a send as finished
            //
            else if (completion->opcode==IBV_WC_RECV && completion->byte_len==0) {
                uint32_t tag = completion->imm_data;
                LOG_DEBUG_MSG("zero byte receive with tag " << hexuint32(tag));

                // bookkeeping : decrement counter that keeps preposted queue full
                client->popReceive();

                // let go of this region (waste really as this was a zero byte message)
                region = (RdmaMemoryRegion *)completion->wr_id;
                chunk_pool_->deallocate(region);

                // now release any zero copy regions we were holding until parcel complete
                active_send_iterator current_send;
                {
                    hpx::lcos::local::spinlock::scoped_lock lk(TagSendCompletionMap_mutex);
                    send_wr_map::iterator it = TagSendCompletionMap.find(tag);
                    if (it==TagSendCompletionMap.end()) {
                        LOG_ERROR_MSG("Tag not present in Send map, FATAL");
                        std::terminate();
                    }
                    current_send = it->second;
                    TagSendCompletionMap.erase(it);
                }
                //
                delete_send_data(current_send);
                //
                _rdmaController->refill_client_receives();
                return 0;
            }
            //
            // When an unmatched receive completes, it is a new parcel, if everything fits into
            // the header, call decode message, otherwise, queue all the Rdma Get operations
            // necessary to complete the message
            //
            else if (completion->opcode==IBV_WC_RECV) {

                util::high_resolution_timer timer;
                LOG_DEBUG_MSG("Entering receive (completion handler) section with received size " << decnumber(completion->byte_len));

                // bookkeeping : decrement counter that keeps preposted queue full
                client->popReceive();

                // get the header of the new message/parcel
                parcel_receive_data r_data;
                r_data.header_region = (RdmaMemoryRegion *)completion->wr_id;;
                header_type *h = (header_type*)r_data.header_region->getAddress();
                LOG_DEBUG_MSG( "received IBV_WC_RECV " <<
                        "buffsize " << decnumber(h->size())
                        << "numbytes " << decnumber(h->numbytes())
                        << "chunks zerocopy( " << decnumber(h->num_chunks().first) << ") "
                        << ", normal( " << decnumber(h->num_chunks().second) << ") "
                        << " chunkdata " << decnumber((h->chunk_data()!=NULL))
                        << " piggyback " << decnumber((h->piggy_back()!=NULL))
                        << " tag " << hexuint32(h->tag())
                );
                // each parcel has a unique tag which we use to organize zero-copy data if we need any
                uint32_t tag = h->tag();

                // when message is fully received, we will hand it to decode_parcel which will release any memory
                // when it destroys the container. Place everything in a struct and put it into a map
                // with a key which is the tag of the parcel
                // Note create entries in map before posting any RDMA reads so that completions don't cause races
                tag_receive_map::iterator it;
                {
                    hpx::lcos::local::spinlock::scoped_lock(TagReceiveCompletionMap_mutex);
                    LOG_DEBUG_MSG("Moving receive data into TagReceiveCompletionMap with tag " << hexuint32(tag));
                    auto ret = TagReceiveCompletionMap.insert(std::make_pair(tag, std::move(r_data)));
                    if (!ret.second) {
                        std::terminate();
                        throw std::runtime_error("Failed to insert tag in TagReceiveCompletionMap");
                    }
                    it = ret.first;
                }
                parcel_receive_data &receive_data = it->second;

                // setting this flag to false disables final parcel receive call as more data is needed
                bool parcel_complete = true;

                char *chunk_data = h->chunk_data();
                if (chunk_data) {
                    // all the info about chunks we need is stored inside the header
                    receive_data.chunks.resize(h->num_chunks().first + h->num_chunks().second);
                    size_t chunkbytes = receive_data.chunks.size() * sizeof(serialization::serialization_chunk);
                    std::memcpy(receive_data.chunks.data(), chunk_data, chunkbytes);
                    LOG_DEBUG_MSG("Copied chunk data from header : size " << decnumber(chunkbytes));

                    // setup info for zero-copy rdma get chunks (if there are any)
                    receive_data.counter = h->num_chunks().first;
                    if (receive_data.counter>0) {
                        parcel_complete = false;
                        int index = 0;
                        for (serialization::serialization_chunk &c : receive_data.chunks) {
                            LOG_DEBUG_MSG("chunk : size " << hexnumber(c.size_)
                                    << " type " << decnumber((uint64_t)c.type_)
                                    << " rkey " << decnumber(c.rkey_)
                                    << " cpos " << hexpointer(c.data_.cpos_)
                                    << " pos " << hexpointer(c.data_.pos_)
                                    << " index " << decnumber(c.data_.index_));
                        }
                        for (serialization::serialization_chunk &c : receive_data.chunks) {
                            if (c.type_ == serialization::chunk_type_pointer) {
                                RdmaMemoryRegion *get_region;
                                if (c.size_<=DEFAULT_MEMORY_POOL_CHUNK_SIZE) {
                                    get_region = chunk_pool_->allocateRegion(std::max(c.size_, (std::size_t)DEFAULT_MEMORY_POOL_CHUNK_SIZE));
                                }
                                else {
                                    get_region = chunk_pool_->AllocateTemporaryBlock(c.size_);
                                }
                                LOG_DEBUG_MSG("RDMA Get address " << hexpointer(c.data_.cpos_)
                                        << " rkey " << decnumber(c.rkey_) << " size " << hexnumber(c.size_)
                                        << " tag " << hexuint32(tag)
                                        << " local address " << get_region->getAddress() << " length " << c.size_);
                                receive_data.zero_copy_regions.push_back(get_region);
                                LOG_DEBUG_MSG("Zero copy regions size is (create) " << receive_data.zero_copy_regions.size());
                                // put region into map before posting read in case it completes whilst this thread is suspended
                                {
                                    hpx::lcos::local::spinlock::scoped_lock lk2(ZeroCopyGetCompletionMap_mutex);
                                    ZeroCopyGetCompletionMap[(uint64_t)get_region] = tag;
                                }
                                // overwrite the serialization data to account for the local pointers instead of remote ones
                                /// post the rdma read/get
                                const void *remoteAddr = c.data_.cpos_;
                                receive_data.chunks[index] = hpx::serialization::create_pointer_chunk(get_region->getAddress(), c.size_, c.rkey_);
                                client->postRead(get_region, c.rkey_, remoteAddr, c.size_);
                            }
                            index++;
                        }
                    }
                }
                else {
                    std::terminate();
                    throw std::runtime_error("@TODO implement RDMA GET of mass chunk information when header too small");
                }

                char *piggy_back = h->piggy_back();
                LOG_DEBUG_MSG("piggy_back is " << hexpointer(piggy_back) << " chunk data is " << hexpointer(h->chunk_data()));
                // if the main serialization chunk is piggybacked in second SGE
                if (piggy_back) {
                    if (parcel_complete) {
                        rcv_data_type wrapped_pointer(piggy_back, h->size(),
                                boost::bind(&parcelport::release_memory_buffers, this, tag));

                        LOG_DEBUG_MSG("calling parcel decode for complete NORMAL parcel");
                        rcv_buffer_type buffer(std::move(wrapped_pointer));
                        buffer.data_.resize(static_cast<std::size_t>(h->size()));
                        decode_message_with_chunks(*this, std::move(buffer), 1, receive_data.chunks);
                        LOG_DEBUG_MSG("parcel decode called for complete NORMAL parcel");
                        //                      chunk_pool_->deallocate(receive_data.header_region);
                    }
                }
                else {
                    std::terminate();
                    throw std::runtime_error("@TODO implement RDMA GET of message when header too small");
                }

                // @TODO replace performance counter data
                //          performance_counters::parcels::data_point& data = buffer.data_point_;
                //          data.time_ = timer.elapsed_nanoseconds();
                //          data.bytes_ = static_cast<std::size_t>(buffer.size_);
                //          ...
                //          data.time_ = timer.elapsed_nanoseconds() - data.time_;
                _rdmaController->refill_client_receives();
                return 0;
            }
            return 0;
        }

        ~parcelport() {
            FUNC_START_DEBUG_MSG;
            _rdmaController = nullptr;
            FUNC_END_DEBUG_MSG;
        }

        /// return true if this pp can be used at bootstrapping, otherwise omit
        bool can_bootstrap() const {
            //    FUNC_START_DEBUG_MSG;
            //    FUNC_END_DEBUG_MSG;
            return false;
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
                if (NULL != sec) {
                    LOG_DEBUG_MSG("Returning some made up agas locality")
                            return
                                    parcelset::locality(
                                            locality(
                                                    _ibv_ip
                                                    , _port
                                            )
                                    );
                }
            }
            FUNC_END_DEBUG_MSG;
            // ibverbs can't be used for bootstrapping
            LOG_DEBUG_MSG("Returning NULL agas locality")
            return parcelset::locality(locality(0xFFFF, 0));
        }

        parcelset::locality create_locality() const {
            FUNC_START_DEBUG_MSG;
            FUNC_END_DEBUG_MSG;
            return parcelset::locality(locality());
        }

        void put_parcels(std::vector<parcelset::locality> dests,
                std::vector<parcel> parcels,
                std::vector<write_handler_type> handlers)
        {
            HPX_ASSERT(dests.size() == parcels.size());
            HPX_ASSERT(dests.size() == handlers.size());
            for(std::size_t i = 0; i != dests.size(); ++i)
            {
                put_parcel(dests[i], parcels[i], handlers[i]);
            }
        }

        void send_early_parcel(parcelset::locality const & dest, parcel& p) {
            FUNC_START_DEBUG_MSG;
            FUNC_END_DEBUG_MSG;
            // Only necessary if your PP an be used at bootstrapping
            put_parcel(dest, p, boost::bind(&parcelport::early_write_handler, this, ::_1, p));
        }

        util::io_service_pool* get_thread_pool(char const* name) {
            FUNC_START_DEBUG_MSG;
            FUNC_END_DEBUG_MSG;
            return 0;
        }

        // This parcelport doesn't maintain a connection cache
        boost::int64_t get_connection_cache_statistics(connection_cache_statistics_type, bool reset) {
            FUNC_START_DEBUG_MSG;
            FUNC_END_DEBUG_MSG;
            return 0;
        }

        void remove_from_connection_cache(parcelset::locality const& loc) {
            FUNC_START_DEBUG_MSG;
            FUNC_END_DEBUG_MSG;
        }

        bool run(bool blocking = true) {
            FUNC_START_DEBUG_MSG;
            _rdmaController->startup();
            LOG_DEBUG_MSG("Fetching memory pool");
            chunk_pool_ = _rdmaController->getMemoryPool();

            LOG_DEBUG_MSG("Setting Connection function");
            auto connection_function = std::bind( &parcelport::handle_verbs_connection, this, std::placeholders::_1, std::placeholders::_2);
            _rdmaController->setConnectionFunction(connection_function);

            LOG_DEBUG_MSG("Setting Completion function");
            // need to use std:bind here as rdmahelper lib uses it too
            auto completion_function = std::bind( &parcelport::handle_verbs_completion, this, std::placeholders::_1, std::placeholders::_2);
            _rdmaController->setCompletionFunction(completion_function);

            FUNC_END_DEBUG_MSG;
            // This should start the receiving side of your PP
            return true;
        }

        void stop(bool blocking = true) {
            FUNC_START_DEBUG_MSG;
            if (!stopped_) {
                bool finished = false;
                do {
                    finished =
                            ZeroCopyGetCompletionMap.empty() &&
                            TagReceiveCompletionMap.empty() &&
                            SendCompletionMap.empty() &&
                            TagSendCompletionMap.empty();
                    if (!finished) {
                        LOG_ERROR_MSG("Entering STOP when not all parcels have completed");
                        std::terminate();
                        do_background_work(1);
                    }
                } while (!finished);

                typedef std::unique_lock<std::mutex>  lock_type2;
                lock_type2 lock(stop_mutex);
                _rdmaController->for_each_client(
                        [](std::pair<const uint32_t, RdmaClientPtr> &map_pair) {
                    if (map_pair.second->getInitiatedConnection()) {
                        _rdmaController->removeServerToServerConnection(map_pair.second);
                    }
                }
                );
                // wait for all clients initiated elsewhere to be disconnected
                while (_rdmaController->num_clients()!=0) {
                    _rdmaController->eventMonitor(0);
                }
                LOG_DEBUG_MSG("wait done");
            }
            stopped_ = true;
            // Stop receiving and sending of parcels
        }

        void enable(bool new_state) {
            // enable/disable sending and receiving of parcels
        }

        // ----------------------------------------------------------------------------------------------
        // called by hpx when an action is invoked on a remote locality.
        // This must be thread safe in order to function as any thread may invoke an action
        // ----------------------------------------------------------------------------------------------
        void put_parcel(parcelset::locality const & dest, parcel p, write_handler_type f) {
            // FUNC_START_DEBUG_MSG;
            boost::uint32_t dest_ip = dest.get<locality>().ip_;
            LOG_DEBUG_MSG("Locality " << ipaddress(_ibv_ip) << " put_parcel to " << ipaddress(dest_ip) );

            // @TODO, don't need smartpointers here, remove them as they waste an atomic refcount
            RdmaClientPtr client;
            {
                // lock this region as we are creating a connection to a remote locality
                // if two threads attempt to do this at the same time, we'll get multiple clients
                boost::lock_guard<hpx::lcos::local::spinlock> lock(connection_mutex);
                ip_map_iterator ip_it = ip_qp_map.find(dest_ip);
                if (ip_it!=ip_qp_map.end()) {
                    LOG_DEBUG_MSG("Connection found with qp " << ip_it->second);
                    client = _rdmaController->getClient(ip_it->second);
                }
                else {
                    LOG_DEBUG_MSG("Connection required to " << ipaddress(dest_ip));
                    client = _rdmaController->makeServerToServerConnection(dest_ip, _rdmaController->getPort());
                    client->setInitiatedConnection(true);
                    LOG_DEBUG_MSG("Setting qpnum in main client map");
                    ip_qp_map[dest_ip] = client->getQpNum();
                }
            }

            // connection ok, we can now send required info to the remote peer
            {
                util::high_resolution_timer timer;

                // the send buffer is created with our allocator and will get memory from our pool
                // - disable deallocation so that we can manage the block lifetime better
                // @TODO, integrate the pointer wrapper and allocators better into parcel_buffer
                allocator_type alloc(*chunk_pool_.get());
                alloc.disable_deallocate = true;
                snd_buffer_type buffer(alloc);

                // encode the parcel directly into an rdma pinned memory block
                // if the serialization overflows the block, panic and rewrite this.
                LOG_DEBUG_MSG("Encoding parcel");
                encode_parcels(&p, std::size_t(-1), buffer, archive_flags_, chunk_pool_->default_chunk_size());
                buffer.data_point_.time_ = timer.elapsed_nanoseconds();

                // create a tag, needs to be unique per client
                uint32_t tag = tag_provider_.next(dest_ip);
                LOG_DEBUG_MSG("Generated tag " << hexuint32(tag) << " from " << hexuint32(dest_ip));

                // we must store details about this parcel so that all memory buffers can be kept
                // until all send operations have completed.
                active_send_iterator current_send;
                {
                    hpx::lcos::local::spinlock::scoped_lock(active_send_mutex);
                    current_send = active_sends.insert(active_sends.end(), parcel_send_data());
                }
                parcel_send_data &send_data = *current_send;
                send_data.tag     = tag;
                send_data.parcel  = p;
                send_data.handler = f;
                send_data.header_region  = NULL;
                send_data.message_region = NULL;
                send_data.chunk_region   = NULL;

                LOG_DEBUG_MSG("Generated unique dest " << hexnumber(dest_ip) << " coded tag " << hexuint32(send_data.tag));

                // for each zerocopy chunk, we must create a memory region for the data
                for (serialization::serialization_chunk &c : buffer.chunks_) {
                    LOG_DEBUG_MSG("chunk : size " << hexnumber(c.size_)
                            << " type " << decnumber((uint64_t)c.type_)
                            << " rkey " << decnumber(c.rkey_)
                            << " cpos " << hexpointer(c.data_.cpos_)
                            << " pos " << hexpointer(c.data_.pos_)
                            << " index " << decnumber(c.data_.index_));
                }
                int index = 0;
                for (serialization::serialization_chunk &c : buffer.chunks_) {
                    if (c.type_ == serialization::chunk_type_pointer) {
                        // if the data chunk fits into a memory block, copy it
                        util::high_resolution_timer regtimer;
                        RdmaMemoryRegion *zero_copy_region;
                        if (c.size_<=HPX_PARCELPORT_VERBS_MEMORY_COPY_THRESHOLD) {
                            zero_copy_region = chunk_pool_->allocateRegion(std::max(c.size_, (std::size_t)DEFAULT_MEMORY_POOL_CHUNK_SIZE));
                            char *zero_copy_memory = (char*)(zero_copy_region->getAddress());
                            std::memcpy(zero_copy_memory, c.data_.cpos_, c.size_);
                            // the pointer in the chunk info must be changed
                            buffer.chunks_[index] = serialization::create_pointer_chunk(zero_copy_memory, c.size_);
                            LOG_DEBUG_MSG("Time to copy memory (ns) " << decnumber(regtimer.elapsed_nanoseconds()));
                        }
                        else {
                            // create a memory region from the pointer
                            zero_copy_region = new RdmaMemoryRegion(
                                    _rdmaController->getProtectionDomain(), c.data_.cpos_, std::max(c.size_, (std::size_t)DEFAULT_MEMORY_POOL_CHUNK_SIZE));
                            LOG_DEBUG_MSG("Time to register memory (ns) " << decnumber(regtimer.elapsed_nanoseconds()));
                        }
                        c.rkey_  = zero_copy_region->getRemoteKey();
                        LOG_DEBUG_MSG("Zero-copy rdma Get region created for address "
                                << hexpointer(zero_copy_region->getAddress())
                                << " and rkey " << decnumber(c.rkey_));
                        send_data.zero_copy_regions.push_back(zero_copy_region);
                    }
                    index++;
                }

                // grab a memory block from the pinned pool to use for the header
                send_data.header_region = chunk_pool_->allocateRegion(chunk_pool_->default_chunk_size());
                char *header_memory = (char*)(send_data.header_region->getAddress());

                // create the header in the pinned memory block
                LOG_DEBUG_MSG("Placement new for the header with piggyback copy disabled");
                header_type *h = new(header_memory) header_type(buffer, send_data.tag, false);
                h->assert_valid();
                send_data.header_region->setMessageLength(h->header_length());
                LOG_DEBUG_MSG(
                        "sending, buffsize " << decnumber(h->size())
                        << "numbytes " << decnumber(h->numbytes())
                        << "chunks zerocopy( " << decnumber(h->num_chunks().first) << ") "
                        << ", normal( " << decnumber(h->num_chunks().second) << ") "
                        << ", chunk_flag " << decnumber(h->header_length())
                        << ", chunk_flag " << decnumber(h->header_length())
                        << "tag " << hexuint32(h->tag())
                );

                // Get the block of pinned memory where the message was encoded during serialization
                // (our allocator was used, so we can find it)
                // @TODO : find a nicer way of handling this block retrieval
                send_data.message_region = chunk_pool_->RegionFromAddress((char*)buffer.data_.data());
                LOG_DEBUG_MSG("Finding region allocated during encode_parcel : address " << hexpointer(buffer.data_.data()) << " region "<< hexpointer(send_data.message_region));
                send_data.message_region->setMessageLength(h->size());

                RdmaMemoryRegion *region_list[] = { send_data.header_region, send_data.message_region };
                int num_regions = 2;
                if (h->chunk_data()) {
                    LOG_DEBUG_MSG("Chunk info is piggybacked");
                    send_data.chunk_region   = NULL;
                }
                else {
                    throw std::runtime_error(
                            "@TODO : implement chunk info rdma get when zero-copy chunks exceed header space");
                }

                if (h->piggy_back()) {
                    LOG_DEBUG_MSG("Main message is piggybacked");
                }
                else {
                    region_list[1] = NULL;
                    num_regions = 1;
                    throw std::runtime_error(
                            "@TODO : implement message rdma get from destination when size exceeds header space");
                }

                uint64_t wr_id = (uint64_t)(send_data.header_region);
                {
                    // add wr_id's to completion map
                    hpx::lcos::local::spinlock::scoped_lock(SendCompletionMap_mutex);
                    if (SendCompletionMap.find(wr_id) != SendCompletionMap.end()) {
                        LOG_ERROR_MSG("wr_id duplicated : FATAL ");
                        std::terminate();
                        throw std::runtime_error("wr_id duplicated in put_parcel : FATAL");
                    }
                    // put everything into map to be retrieved when send completes
                    SendCompletionMap[wr_id] = current_send;
                    LOG_DEBUG_MSG("wr_id for send added to WR completion map "
                            << hexpointer(wr_id) << " Entries " << SendCompletionMap.size());
                }
                {
                    // if there are zero copy regions, we must hold onto them until the destination tells us
                    // it has completed all rdma Get operations
                    if (!send_data.zero_copy_regions.empty()) {
                        hpx::lcos::local::spinlock::scoped_lock lk(TagSendCompletionMap_mutex);
                        // put the data into a new map which is indexed by the Tag of the send
                        // zero copy blocks will be released when we are told this has completed
                        TagSendCompletionMap[send_data.tag] = current_send;
                    }
                }

                // send the header/main_chunk to the destination, wr_id is header_region (entry 0 in region_list)
                client->postSend_xN(region_list, 2, true, false, 0);
                LOG_TRACE_MSG("Block header_region"
                        << " region "    << hexpointer(send_data.header_region)
                        << " buffer "    << hexpointer(send_data.header_region->getAddress()));
                LOG_TRACE_MSG("Block message_region"
                        << " region "    << hexpointer(send_data.message_region)
                        << " buffer "    << hexpointer(send_data.message_region->getAddress()));


                // log the time spent in performance counter
                buffer.data_point_.time_ =
                        timer.elapsed_nanoseconds() - buffer.data_point_.time_;

                // parcels_sent_.add_data(buffer.data_point_);
            }
            FUNC_END_DEBUG_MSG;
        }

        mutable mutex_type background_mtx;

        // ----------------------------------------------------------------------------------------------
        // This is called whenever a HPX OS thread is idling, can be used to poll for incoming messages
        // this should be thread safe as eventMonitor only polls and dispatches and is thread safe.
        // ----------------------------------------------------------------------------------------------
        bool do_background_work(std::size_t num_thread) {
            if (stopped_)
                return false;
            //        mutex_type::scoped_lock mtx(background_mtx);
            // if an event comes in, we may spend time processing/handling it and another may arrive
            // during this handling, so keep checking until none are received
            bool done = false;
            do {
                done = (_rdmaController->eventMonitor(0) == 0);
            } while (!done);
            return true;
        }

    private:

        // ----------------------------------------------------------------------------------------------
        // Only needed for bootstrapping
        void early_write_handler(boost::system::error_code const& ec, parcel const & p) {
            FUNC_START_DEBUG_MSG;
            if (ec) {
                // all errors during early parcel handling are fatal
                boost::exception_ptr exception = hpx::detail::get_exception(hpx::exception(ec), "mpi::early_write_handler",
                        __FILE__, __LINE__,
                        "error while handling early parcel: " + ec.message() + "(" + boost::lexical_cast < std::string
                        > (ec.value()) + ")" + parcelset::dump_parcel(p));

                hpx::report_error(exception);
            }
            FUNC_END_DEBUG_MSG;
        }
    };
}
}
}
}

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
            std::cout << "init logging 1 in plugin start " << std::endl;
#ifndef RDMAHELPER_DISABLE_LOGGING 
            std::cout << "init logging 2 in plugin start " << std::endl;
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
                "ifname = ${HPX_PARCELPORT_VERBS_IFNAME:" HPX_PARCELPORT_VERBS_IFNAME "}\n"
                "device = ${HPX_PARCELPORT_VERBS_DEVICE:" HPX_PARCELPORT_VERBS_DEVICE "}\n"
                "interface = ${HPX_PARCELPORT_VERBS_INTERFACE:" HPX_PARCELPORT_VERBS_INTERFACE "}\n"
                "memory_chunk_size = ${HPX_PARCEL_IBVERBS_MEMORY_CHUNK_SIZE:"
                BOOST_PP_STRINGIZE(HPX_PARCELPORT_VERBS_MEMORY_CHUNK_SIZE) "}\n"
        "max_memory_chunks = ${HPX_PARCEL_IBVERBS_MAX_MEMORY_CHUNKS:"
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
std::string       hpx::parcelset::policies::verbs::parcelport::_ibverbs_ifname;
std::string       hpx::parcelset::policies::verbs::parcelport::_ibverbs_device;
std::string       hpx::parcelset::policies::verbs::parcelport::_ibverbs_interface;
boost::uint32_t   hpx::parcelset::policies::verbs::parcelport::_ibv_ip;
boost::uint32_t   hpx::parcelset::policies::verbs::parcelport::_port;


HPX_REGISTER_PARCELPORT(hpx::parcelset::policies::verbs::parcelport, verbs);
