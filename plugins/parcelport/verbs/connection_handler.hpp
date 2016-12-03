//  Copyright (c) 2015 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_CONNECTION_HANDLER_HPP
#define HPX_PARCELSET_POLICIES_CONNECTION_HANDLER_HPP

#include <hpx/config/warnings_prefix.hpp>
#include <hpx/runtime/parcelset/locality.hpp>
#include <hpx/runtime/parcelset/parcelport_impl.hpp>
//
#include <plugins/parcelport/verbs/rdma/rdma_logging.hpp>
#include <plugins/parcelport/verbs/rdma/rdma_error.hpp>
#include "rdma/verbs_device.hpp"
//
namespace hpx {
namespace parcelset {
namespace policies {
namespace verbs
{
    class HPX_EXPORT parcelport;
}}

    template <>
    struct connection_handler_traits<policies::verbs::parcelport>
    {
        typedef policies::verbs::sender_connection      connection_type;
        typedef HPX_PARCELPORT_VERBS_HAVE_BOOTSTRAPPING send_early_parcel;
        typedef std::true_type                          do_background_work;
        typedef std::true_type                          send_immediate_parcels;

        static const char * type()
        {
            return "verbs";
        }

        static const char * pool_name()
        {
            return "parcel-pool-verbs";
        }

        static const char * pool_name_postfix()
        {
            return "-verbs";
        }
    };

namespace policies {
namespace verbs
{
        uint32_t get_verbs_device_address(const char *devicename, const char *iface, char *hostname)
        {
          FUNC_START_DEBUG_MSG
          // Find the address of the I/O link device.
          verbs_device_ptr linkDevice;
          try {
            linkDevice = verbs_device_ptr(new verbs_device(devicename, iface));
          }
          catch (rdma_error& e) {
            LOG_ERROR_MSG("error opening InfiniBand device: " << e.what());
          }
          LOG_DEBUG_MSG("Created InfiniBand device for " << linkDevice->get_device_name() << " using interface " << linkDevice->get_interface_name());

          std::stringstream temp;
          in_addr_t addr = linkDevice->get_address();
          temp
            << (int)((uint8_t*)&addr)[0] << "."
            << (int)((uint8_t*)&addr)[1] << "."
            << (int)((uint8_t*)&addr)[2] << "."
            << (int)((uint8_t*)&addr)[3] << std::ends;
          strcpy(hostname, temp.str().c_str());
          //
          LOG_DEBUG_MSG("Generated hostname string " << hostname);

          // print device info for debugging
        //  linkDevice->getDeviceInfo(true);
          FUNC_END_DEBUG_MSG
          return (uint32_t)(addr);
        }

}}}}

#include <hpx/config/warnings_suffix.hpp>

#endif
