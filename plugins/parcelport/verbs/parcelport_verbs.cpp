//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config/defines.hpp>

#include <hpx/hpx_fwd.hpp>

#include <hpx/plugins/parcelport_factory.hpp>
#include <hpx/util/command_line_handling.hpp>

// parcelport
#include <hpx/runtime.hpp>
#include <hpx/runtime/parcelset/parcelport.hpp>
#include <hpx/runtime/parcelset/parcel_buffer.hpp>
#include <hpx/runtime/parcelset/encode_parcels.hpp>
// Local parcelport plugin
#include <connection_handler_verbs.hpp>

#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/safe_lexical_cast.hpp>

// rdmahelper library
#include "RdmaLogging.h"
#include "RdmaController.h"
#include "RdmaDevice.h"

namespace hpx { namespace parcelset { namespace policies { namespace verbs {

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

  void save(util::portable_binary_oarchive & ar) const {
    // save the state
    FUNC_START_DEBUG_MSG;
    ar.save(ip_);
    FUNC_END_DEBUG_MSG;
  }

  void load(util::portable_binary_iarchive & ar) {
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

    FUNC_END_DEBUG_MSG;
    return parcelset::locality(locality(_ibv_ip, _ibv_ip));
  }

public:
  parcelport(util::runtime_configuration const& ini,
      util::function_nonser<void(std::size_t, char const*)> const& on_start_thread,
      util::function_nonser<void()> const& on_stop_thread) :
      parcelset::parcelport(ini, here(ini), "verbs"), archive_flags_(boost::archive::no_header)
  {
    FUNC_START_DEBUG_MSG;
    _port   = 0;
#ifdef BOOST_BIG_ENDIAN
    std::string endian_out = get_config_entry("hpx.parcel.endian_out", "big");
#else
    std::string
    endian_out = get_config_entry("hpx.parcel.endian_out", "little");
#endif
    if (endian_out == "little")
      archive_flags_ |= util::endian_little;
    else if (endian_out == "big")
      archive_flags_ |= util::endian_big;
    else {
      HPX_ASSERT(endian_out == "little" || endian_out == "big");
    }

    if (!this->allow_array_optimizations()) {
      archive_flags_ |= util::disable_array_optimization;
      archive_flags_ |= util::disable_data_chunking;
      std::cout << "Disabling array optimization and data chunking " << std::endl;
    } else {
      if (!this->allow_zero_copy_optimizations()) {
        archive_flags_ |= util::disable_data_chunking;
        std::cout << "Disabling data chunking " << std::endl;
      }
    }
    _rdmaController = std::make_shared<RdmaController>(_ibverbs_device.c_str(), _ibverbs_interface.c_str(), 0);

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

  void put_parcels(std::vector<parcelset::locality> dests, std::vector<parcel> parcels,
      std::vector<write_handler_type> handlers) {
    FUNC_START_DEBUG_MSG;
    FUNC_END_DEBUG_MSG;
    // Put a vector of parcels
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
    FUNC_END_DEBUG_MSG;
    // This should start the receiving side of your PP
    return true;
  }

  void stop(bool blocking = true) {
    FUNC_START_DEBUG_MSG;
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
    ip_map_iterator ip_it = ip_qp_map.find(dest_ip);
    if (ip_it!=ip_qp_map.end()) {
      LOG_DEBUG_MSG("Connection found with qp " << ip_it->second);
    }
    else {
      LOG_DEBUG_MSG("Connection required to " << ipaddress(dest_ip));
      RdmaClientPtr client = _rdmaController->makeServerToServerConnection(dest_ip, _rdmaController->getPort());
      ip_qp_map[dest_ip] = client->getQpNum();
    }
    // Send a single parcel, after succesful sending, f should be called.
    FUNC_END_DEBUG_MSG;
  }

  bool do_background_work(std::size_t num_thread) {
//    FUNC_START_DEBUG_MSG;
    _rdmaController->eventMonitor(0);
//    FUNC_END_DEBUG_MSG;
    // This is called whenever a HPX OS thÍread is idling, can be used to poll for incoming messages
    return true;
  }

private:
  int archive_flags_;

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

  LOG_DEBUG_MSG("\n"
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
      "enable = 0");
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
