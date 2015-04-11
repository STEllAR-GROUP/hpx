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

#define HPX_PARCELPORT_VERBS_MEMORY_COPY_THRESHOLD DEFAULT_MEMORY_POOL_CHUNK_SIZE

using namespace hpx::parcelset::policies;

namespace hpx { namespace parcelset { namespace policies { namespace verbs
{
/*
    struct rdma_serialization_chunk
    {
        serialization::chunk_data   data_;       // index or pointer
        std::size_t                 size_;       // size of the serialization_chunk starting at index_/pos_
        boost::uint8_t              type_;       // chunk_type
        boost::uint32_t             remote_key;  // RDMA remote key that will be used for put/get
        boost::uint64_t             remote_addr; // RDMA address that will be used by remote put/get
        rdma_serialization_chunk(const serialization::serialization_chunk &other)
            data_(other.data_), size_(other.size_), type_(other.type_), remote_key(0), remote_addr(0) {}

    };
*/
    struct locality {
      static const char *type() {
        FUNC_START_DEBUG_MSG;
        FUNC_END_DEBUG_MSG;
        return "verbs";
      }

      explicit locality(boost::uint32_t ip, boost::uint32_t port) :
            ip_(ip), port_(port), qp_(0xFFFF) {}

      locality() : ip_(0xFFFF), port_(0), qp_(0xFFFF) {}

      // some condition marking this locality as valid
      operator util::safe_bool<locality>::result_type() const {
        //    FUNC_START_DEBUG_MSG;
        //    FUNC_END_DEBUG_MSG;
        return util::safe_bool<locality>()(ip_ != boost::uint32_t(0xFFFF));
      }

      void save(serialization::output_archive & ar) const {
        // save the state
        FUNC_START_DEBUG_MSG;
        ar.save(ip_);
        FUNC_END_DEBUG_MSG;
      }

      void load(serialization::input_archive & ar) {
        // load the state
        FUNC_START_DEBUG_MSG;
        ar.load(ip_);
        FUNC_END_DEBUG_MSG;
      }

    private:
      friend bool operator==(locality const & lhs, locality const & rhs) {
        FUNC_START_DEBUG_MSG;
        FUNC_END_DEBUG_MSG;
        return lhs.ip_ == rhs.ip_;
      }

      friend bool operator<(locality const & lhs, locality const & rhs) {
        FUNC_START_DEBUG_MSG;
        FUNC_END_DEBUG_MSG;
        return lhs.ip_ < rhs.ip_;
      }

      friend std::ostream & operator<<(std::ostream & os, locality const & loc) {
        FUNC_START_DEBUG_MSG;
        boost::io::ios_flags_saver
        ifs(os);
        os << loc.ip_;
        FUNC_END_DEBUG_MSG
        return os;
      }
    public:
      boost::uint32_t ip_;
      boost::uint32_t port_;
      boost::uint32_t qp_;
    };

    class parcelport: public parcelset::parcelport {
    private:
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
              char buffty[256];
              _ibv_ip = hpx::parcelset::policies::verbs::Get_rdma_device_address(_ibverbs_device.c_str(), _ibverbs_interface.c_str(), buffty);
              LOG_DEBUG_MSG("here() got hostname of " << buffty);
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
      parcelport(util::runtime_configuration const& ini,
          util::function_nonser<void(std::size_t, char const*)> const& on_start_thread,
          util::function_nonser<void()> const& on_stop_thread) :
            parcelset::parcelport(ini, here(ini), "verbs"), archive_flags_(0) // boost::archive::no_header)
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
//            archive_flags_ |= serialization::disable_data_chunking;
//            LOG_DEBUG_MSG("Disabling data chunking");
          }
        }
        _rdmaController = std::make_shared<RdmaController>(_ibverbs_device.c_str(), _ibverbs_interface.c_str(), _port);

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

      typedef header<DEFAULT_MEMORY_POOL_CHUNK_SIZE>                    header_type;
      typedef hpx::lcos::local::spinlock                                mutex_type;
      typedef char                                                      memory_type;
      typedef RdmaMemoryPool                                            memory_pool_type;
      typedef std::shared_ptr<memory_pool_type>                         memory_pool_ptr_type;
      typedef hpx::util::detail::memory_chunk_pool_allocator
          <memory_type, memory_pool_type, mutex_type>                   allocator_type;
      typedef std::vector<memory_type, allocator_type>                  snd_data_type;
      typedef util::detail::pointer_wrapper_vector<memory_type>         rcv_data_type;
      typedef parcel_buffer<snd_data_type>                              snd_buffer_type;
      typedef parcel_buffer<rcv_data_type, std::vector<memory_type>>    rcv_buffer_type;
      //
      int                       archive_flags_;
      boost::atomic<bool>       stopped_;
      memory_pool_ptr_type      chunk_pool_;
      tag_provider              tag_provider_;

//      performance_counters::parcels::gatherer& parcels_sent_;

      typedef std::vector<RdmaMemoryRegion*> region_vector;
      typedef std::tuple<parcelset::parcel, parcelset::parcelhandler::write_handler_type, RdmaMemoryRegion*, region_vector> handler_tuple;
      typedef std::unordered_map<uint64_t, handler_tuple> send_wr_map;

      typedef std::pair<parcelset::parcel, parcelset::parcelhandler::write_handler_type> handler_pair;
      typedef std::unordered_map<uint64_t, int> get_wr_map;

      // store na_verbs_op_id objects using a map referenced by verbs work request ID
      send_wr_map SendCompletionMap;
      get_wr_map  ZeroCopyGetCompletionMap;

      // pointer to verbs completion member function type
      typedef int (parcelport::*parcel_member_function)(struct ibv_wc *completion, RdmaClientPtr);

      // handler for completions, this is triggered as a callback from the rdmaController when
      // a completion event has occurred
      int handle_verbs_completion(struct ibv_wc *completion, RdmaClientPtr client)
      {
        LOG_DEBUG_MSG("Completion yay!!!");
        uint64_t wr_id = completion->wr_id;
        RdmaMemoryRegion *region = (RdmaMemoryRegion *)completion->wr_id;
        LOG_DEBUG_MSG("completion wr_id " << hexpointer(region) << " address " << hexpointer(region->getAddress()));
        //
        send_wr_map::iterator it_s;
        get_wr_map::iterator  it_z;
        if ((it_s = SendCompletionMap.find(wr_id)) != SendCompletionMap.end()) {
          LOG_DEBUG_MSG("Entering send (completion handler) section");
          // get the chunk region used in 2nd scatter gather entry (do this before erasing iterator)
          RdmaMemoryRegion *message_region = (RdmaMemoryRegion*)(std::get<2>(it_s->second));

          // a send has completed successfully call the write_handler
          LOG_DEBUG_MSG("Calling write_handler for completed send");
          error_code ec;
          std::get<1>(it_s->second).operator()(ec, std::get<0>(it_s->second));

          // release iterator before memory block to avoid race on memory region reuse
          LOG_DEBUG_MSG("erasing iterator for completed send");
          SendCompletionMap.erase(it_s);

          LOG_DEBUG_MSG("deallocating region 1 for completed send " << hexpointer(region));
          chunk_pool_->deallocate(region);
          // if this message had multiple (2) SGEs then release other regions
          if (message_region) {
            LOG_DEBUG_MSG("deallocating region 2 for completed send " << hexpointer(message_region));
            chunk_pool_->deallocate(message_region);
          }
          return 0;
        }
        if ((it_z = ZeroCopyGetCompletionMap.find(wr_id)) != ZeroCopyGetCompletionMap.end()) {
          LOG_ERROR_MSG("A zero copy get has completed, implement this");
        }
        else {
          // this was an unexpected receive, i.e a new parcel
          LOG_DEBUG_MSG("Entering receive (completion handler) section with received size " << decnumber(completion->byte_len));

          // decrement counter that keeps preposted queue full
          client->popReceive();
          //
          util::high_resolution_timer timer;
          // get the header
          RdmaMemoryRegion *header_region = region;
          header_type *h = (header_type*)header_region->getAddress();

          LOG_DEBUG_MSG( "received " <<
                 "buffer size is " << decnumber(h->size())
              << "numbytes is " << decnumber(h->numbytes())
              << "num_chunks is zerocopy( " << decnumber(h->num_chunks().first) << ") "
              << ", normal( " << decnumber(h->num_chunks().second) << ") "
              << " chunkdata " << decnumber((h->chunk_data()!=NULL))
              << " piggyback " << decnumber((h->piggy_back()!=NULL))
              << " tag is " << decnumber(h->tag())
              );

          // setting this flag to false disables final parcel receive call which might be deferred until
          // more chunks are collected
          bool parcel_complete = true;

          // when the parcel is complete, we will need to release one or more regions
          // store them here - if zero-copy regions are required we must keep them
          // until all operations are done
          std::vector<RdmaMemoryRegion*> regions;
          regions.push_back(header_region);

          std::vector<serialization::serialization_chunk> chunks;
          char *chunk_data = h->chunk_data();
          if (chunk_data) {
              int numchunks = h->num_chunks().first + h->num_chunks().second;
              chunks.resize(numchunks);
              size_t chunkbytes = numchunks * sizeof(serialization::serialization_chunk);
              std::memcpy(chunks.data(), chunk_data, chunkbytes);
              LOG_DEBUG_MSG("Copied chunk data from header : size " << decnumber(chunkbytes));
              BOOST_FOREACH(serialization::serialization_chunk &c, chunks) {
                  if (c.type_ == serialization::chunk_type_pointer) {
                      LOG_DEBUG_MSG("RDMA Get from address " << hexpointer(c.data_.cpos_) << " with rkey " << decnumber(c.rkey_) << " size " << hexnumber(c.size_));
                      RdmaMemoryRegion *get_region = chunk_pool_->allocateRegion(c.size_);
                      uint64_t wr_id = client->postRead(get_region, c.rkey_, c.data_.cpos_, c.size_);
                      regions.push_back(get_region);
                      ZeroCopyGetCompletionMap[wr_id] = 45; // std::make_tuple(regions);
                      parcel_complete = false;
                  }
              }
          }
          else {
              throw std::runtime_error("@TODO implement RDMA GET of mass chunk information when header too small");
          }

          char *piggy_back = h->piggy_back();
          LOG_DEBUG_MSG("piggy_back is " << hexpointer(piggy_back) << " chunk data is " << hexpointer(h->chunk_data()));
          // if the main serialization chunk is piggybacked in second SGE
          if (piggy_back) {
              rcv_data_type wrapped_pointer(piggy_back, h->size());
              rcv_buffer_type buffer(wrapped_pointer);
              LOG_DEBUG_MSG("Calling buffer resize (should have no effect on our wrapped pointer vector)");
              buffer.data_.resize(static_cast<std::size_t>(h->size()));
              LOG_DEBUG_MSG("Called buffer resize");
              int tag = h->tag();

              if (parcel_complete) {
                decode_message_with_chunks(*this, std::move(buffer), 1, chunks);
                LOG_DEBUG_MSG("parcel decode called for complete parcel");
                chunk_pool_->deallocate(header_region);
              }
          }

          // @TODO replace performance counter data
//          performance_counters::parcels::data_point& data = buffer.data_point_;
//          data.time_ = timer.elapsed_nanoseconds();
//          data.bytes_ = static_cast<std::size_t>(buffer.size_);
//          ...
//          data.time_ = timer.elapsed_nanoseconds() - data.time_;
        }
        _rdmaController->refill_client_receives();
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
            stopped_ = true;
        FUNC_END_DEBUG_MSG;
        // Stop receiving and sending of parcels
      }

      void enable(bool new_state) {
        FUNC_START_DEBUG_MSG;
        FUNC_END_DEBUG_MSG;
        // enable/disable sending and receiving of parcels
      }

      void put_parcel(parcelset::locality const & dest, parcel p, write_handler_type f) {
        FUNC_START_DEBUG_MSG;
        boost::uint32_t dest_ip = dest.get<locality>().ip_;
        LOG_DEBUG_MSG("Locality " << ipaddress(_ibv_ip) << " Sending packet to " << ipaddress(dest_ip) );
        //
        RdmaClientPtr client;
        ip_map_iterator ip_it = ip_qp_map.find(dest_ip);
        if (ip_it!=ip_qp_map.end()) {
          LOG_DEBUG_MSG("Connection found with qp " << ip_it->second);
          client = _rdmaController->getClient(ip_it->second);
        }
        else {
          LOG_DEBUG_MSG("Connection required to " << ipaddress(dest_ip));
          client = _rdmaController->makeServerToServerConnection(dest_ip, _rdmaController->getPort());
          ip_qp_map[dest_ip] = client->getQpNum();
        }

        // connection ok, we can now send required info to the remote peer
        {
          util::high_resolution_timer timer;

          // the send buffer is created with our allocator and will get memory from our pool
          // - disable deallocation so that we can manage the block lifetime
          allocator_type alloc(*chunk_pool_.get());
          alloc.disable_deallocate = true;
          snd_buffer_type buffer(alloc);

          // encode the parcel directly into an rdma pinned memory block
          // if the serialization overflows the block, panic and rewrite this.
          LOG_DEBUG_MSG("Encoding parcel");
          encode_parcels(&p, std::size_t(-1), buffer, archive_flags_, chunk_pool_->default_chunk_size());
          buffer.data_point_.time_ = timer.elapsed_nanoseconds();

          // create a tag (not used at the moment)
          tag_provider::tag tag(tag_provider_());
          LOG_DEBUG_MSG("Tag generated is " << tag.tag_);

          // for each zerocopy chunk, we must create a memory region for the data
          std::vector<RdmaMemoryRegion*> zero_copy_regions;
          BOOST_FOREACH(serialization::serialization_chunk &c, buffer.chunks_) {
              if (c.type_ == serialization::chunk_type_pointer) {
                  // if the data chunk fits into a memory block, copy it
                  util::high_resolution_timer regtimer;
                  RdmaMemoryRegion *zero_copy_region;
                  if (c.size_<=HPX_PARCELPORT_VERBS_MEMORY_COPY_THRESHOLD) {
                      zero_copy_region = chunk_pool_->allocateRegion(c.size_);
                      char *zero_copy_memory = (char*)(zero_copy_region->getAddress());
                      std::memcpy(zero_copy_memory, c.data_.cpos_, c.size_);
                      LOG_DEBUG_MSG("Time to copy memory (ns) " << decnumber(regtimer.elapsed_nanoseconds()));
                      c.rkey_  = 0;
                  }
                  else {
                      // create a memory region from the pointer
                      zero_copy_region = new RdmaMemoryRegion(
                              _rdmaController->getProtectionDomain(),
                              c.data_.cpos_, c.size_);
                      c.rkey_  = zero_copy_region->getRemoteKey();
                      LOG_DEBUG_MSG("remote key is " << decnumber(c.rkey_));
                      LOG_DEBUG_MSG("Time to register memory (ns) " << decnumber(regtimer.elapsed_nanoseconds()));
                  }
                  zero_copy_regions.push_back(zero_copy_region);
              }
          }

          // grab a memory block from the pinned pool to use for the header
          RdmaMemoryRegion *header_region = chunk_pool_->allocateRegion(chunk_pool_->default_chunk_size());
          char *header_memory = (char*)(header_region->getAddress());

          // create the header in the pinned memory block
          LOG_DEBUG_MSG("Placement new for the header with piggyback copy disabled");
          header_type *h = new(header_memory) header_type(buffer, tag, false);
          h->assert_valid();
          header_region->setMessageLength(h->header_length());
          LOG_DEBUG_MSG(
                 "buffer size is " << decnumber(h->size())
              << "numbytes is " << decnumber(h->numbytes())
              << "num_chunks is zerocopy( " << decnumber(h->num_chunks().first) << ") "
              << ", normal( " << decnumber(h->num_chunks().second) << ") "
              << ", chunk_flag " << decnumber(h->header_length())
              << ", chunk_flag " << decnumber(h->header_length())
              << "tag is " << decnumber(h->tag())
              );

          // Get the block of pinned memory where the message was encoded during serialization
          // (our allocator was used, so we can find it)
          RdmaMemoryRegion *message_region = chunk_pool_->RegionFromAddress((char*)buffer.data_.data());
          LOG_DEBUG_MSG("Finding region allocated during encode_parcel " << hexpointer(message_region));
          message_region->setMessageLength(h->size());

          RdmaMemoryRegion *region_list[] = { header_region, message_region };
          int num_regions = 2;
          if (h->chunk_data()) {
              LOG_DEBUG_MSG("Chunk info is piggybacked");
          }
          else {
              throw std::runtime_error(
                  "@TODO : implement chunk info get from destination when num zero-copy chunks is very large");
          }

          if (h->piggy_back()) {
              LOG_DEBUG_MSG("Main message is piggybacked");
          }
          else {
              region_list[1] = NULL;
              num_regions = 1;
              LOG_ERROR_MSG("@TODO : implement message get from destination when size large");
          }

          // send the header/main_chunk to the destination, add wr_id's to completion map
          uint64_t wr_id = client->postSend_xN(region_list, 2, true, false, 0);
          LOG_TRACE_MSG("Block header_region"
              << " region "    << hexpointer(header_region)
              << " buffer "    << hexpointer(header_region->getAddress()));
          LOG_TRACE_MSG("Block message_region"
              << " region "    << hexpointer(message_region)
              << " buffer "    << hexpointer(message_region->getAddress()));
          SendCompletionMap[wr_id] = std::make_tuple(p, f, message_region, std::move(zero_copy_regions));
          LOG_DEBUG_MSG("wr_id for send added to WR completion map "
              << hexpointer(wr_id) << " Entries " << SendCompletionMap.size());

          // log the time spent in performance counter
          buffer.data_point_.time_ =
              timer.elapsed_nanoseconds() - buffer.data_point_.time_;
//          parcels_sent_.add_data(buffer.data_point_);

        }
        FUNC_END_DEBUG_MSG;
      }

      // This is called whenever a HPX OS thread is idling, can be used to poll for incoming messages
      bool do_background_work(std::size_t num_thread) {
        if (stopped_)
          return false;
        // if an event comes in, we may spend time processing/handling it and another may arrive
        // during this handling, so keep checking until non are received
        bool done = false;
        do {
          done = (_rdmaController->eventMonitor(0) == 0);
        } while (!done);
        return true;
      }

    private:

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
      initLogging();
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
    "zero_copy_optimization = 0\n"
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
