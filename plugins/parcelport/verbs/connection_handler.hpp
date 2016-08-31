//  Copyright (c) 2015 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_VERBS_CONNECTION_HANDLER_HPP
#define HPX_PARCELSET_POLICIES_VERBS_CONNECTION_HANDLER_HPP

#include <hpx/config/warnings_prefix.hpp>
#include <hpx/runtime/parcelset/locality.hpp>
#include <hpx/runtime/parcelset/parcelport_impl.hpp>
//
#include <boost/shared_ptr.hpp>
//
#include <RdmaLogging.h>
#include <RdmaError.h>
#include <RdmaDevice.h>

namespace hpx { namespace parcelset
{
    namespace policies { namespace verbs
    {
        class HPX_EXPORT parcelport;
    }}

    template <>
    struct connection_handler_traits<policies::verbs::parcelport>
    {
        typedef policies::verbs::sender_connection connection_type;
        typedef std::true_type  send_early_parcel;
        typedef std::true_type   do_background_work;
        typedef std::false_type  use_connection_cache;

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

    namespace policies { namespace verbs
    {
        uint32_t Get_rdma_device_address(const char *devicename, const char *iface, char *hostname)
        {
          FUNC_START_DEBUG_MSG
        #ifndef __BGQ__
          // Find the address of the I/O link device.
          bgcios::RdmaDevicePtr linkDevice;
          try {
            linkDevice = bgcios::RdmaDevicePtr(new bgcios::RdmaDevice(devicename, iface));
          }
          catch (bgcios::RdmaError& e) {
            LOG_ERROR_MSG("error opening InfiniBand device: " << e.what());
          }
          LOG_DEBUG_MSG("Created InfiniBand device for " << linkDevice->getDeviceName() << " using interface " << linkDevice->getInterfaceName());

          std::stringstream temp;
          in_addr_t addr = linkDevice->getAddress();
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
        #else
          strcpy(hostname,"");
        #endif
          FUNC_END_DEBUG_MSG
          return (uint32_t)(addr);
        }

    }}

}}

#include <hpx/config/warnings_suffix.hpp>

#endif
