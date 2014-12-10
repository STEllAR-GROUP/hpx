//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2013-2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This is needed to make everything work with the Intel MPI library header
#include <hpx/config/defines.hpp>
#if defined(HPX_PARCELPORT_MPI)
#include <mpi.h>
#endif

#include <hpx/hpx_fwd.hpp>

#if defined(HPX_PARCELPORT_TCP)
#include <hpx/runtime/parcelset/policies/tcp/connection_handler.hpp>
#include <hpx/runtime/parcelset/policies/tcp/receiver.hpp>
#include <hpx/runtime/parcelset/policies/tcp/sender.hpp>
#endif
#if defined(HPX_PARCELPORT_IPC)
#include <hpx/runtime/parcelset/policies/ipc/connection_handler.hpp>
#include <hpx/runtime/parcelset/policies/ipc/receiver.hpp>
#include <hpx/runtime/parcelset/policies/ipc/sender.hpp>
#endif
#if defined(HPX_PARCELPORT_IBVERBS)
#include <hpx/runtime/parcelset/policies/ibverbs/connection_handler.hpp>
#include <hpx/runtime/parcelset/policies/ibverbs/receiver.hpp>
#include <hpx/runtime/parcelset/policies/ibverbs/sender.hpp>
#endif
#if defined(HPX_PARCELPORT_MPI)
#include <hpx/runtime/parcelset/policies/mpi/connection_handler.hpp>
#include <hpx/runtime/parcelset/policies/mpi/receiver.hpp>
#include <hpx/runtime/parcelset/policies/mpi/sender.hpp>
#endif

#include <hpx/runtime/parcelset/parcelport_impl.hpp>
#include <hpx/util/io_service_pool.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/safe_lexical_cast.hpp>
#include <hpx/exception.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

namespace hpx { namespace parcelset
{
    /// load the runtime configuration parameters
    std::pair<std::vector<std::string>, bool> parcelport::runtime_configuration(int type)
    {
        typedef std::pair<std::vector<std::string>, bool> return_type;
        switch(type) {
        case connection_tcp:
#if defined(HPX_PARCELPORT_TCP)
            return return_type(
                policies::tcp::connection_handler::runtime_configuration()
              , true);
#endif
        case connection_ipc:
#if defined(HPX_PARCELPORT_IPC)
            return return_type(
                policies::ipc::connection_handler::runtime_configuration()
              , false);
#endif
            break;
        case connection_ibverbs:
#if defined(HPX_PARCELPORT_IBVERBS)
            return return_type(
                policies::ibverbs::connection_handler::runtime_configuration()
              , false);
#endif
            break;

        case connection_portals4:
            break;

        case connection_mpi:
#if defined(HPX_PARCELPORT_MPI)
            return return_type(
                policies::mpi::connection_handler::runtime_configuration()
              , true);
#endif

            break;
        default:
            break;
        }

        return return_type(std::vector<std::string>(), false);
    }

    boost::shared_ptr<parcelport> parcelport::create(int type,
        util::runtime_configuration const& cfg,
        HPX_STD_FUNCTION<void(std::size_t, char const*)> const& on_start_thread,
        HPX_STD_FUNCTION<void()> const& on_stop_thread)
    {
        switch(type) {
        case connection_tcp:
            {
#if defined(HPX_PARCELPORT_TCP)
                if (hpx::util::get_entry_as<int>(cfg, "hpx.parcel.tcp.enable", "1"))
                {
                    return boost::make_shared<policies::tcp::connection_handler>(
                        cfg, on_start_thread, on_stop_thread);
                }
#endif

                HPX_THROW_EXCEPTION(bad_parameter, "parcelport::create",
                    "unsupported connection type 'connection_tcp'");
            }

        case connection_ipc:
            {
#if defined(HPX_PARCELPORT_IPC)
                // Create ipc based parcelport only if allowed by the
                // configuration info.
                if (hpx::util::get_entry_as<int>(cfg, "hpx.parcel.ipc.enable", "0"))
                {
                    return boost::make_shared<policies::ipc::connection_handler>(
                        cfg, on_start_thread, on_stop_thread);
                }
#endif
                HPX_THROW_EXCEPTION(bad_parameter, "parcelport::create",
                    "unsupported connection type 'connection_ipc'");
            }
            break;

        case connection_ibverbs:
#if defined(HPX_PARCELPORT_IBVERBS)
            {
                // Create ibverbs based parcelport only if allowed by the
                // configuration info.
                if (hpx::util::get_entry_as<int>(cfg, "hpx.parcel.ibverbs.enable", "0"))
                {
                    return boost::make_shared<policies::ibverbs::connection_handler>(
                        cfg, on_start_thread, on_stop_thread);
                }
            }
#endif
            HPX_THROW_EXCEPTION(bad_parameter, "parcelport::create",
                "unsupported connection type 'connection_ibverbs'");
            break;

        case connection_portals4:
            HPX_THROW_EXCEPTION(bad_parameter, "parcelport::create",
                "unsupported connection type 'connection_portals4'");
            break;

        case connection_mpi:
#if defined(HPX_PARCELPORT_MPI)
            {
                // Create MPI based parcelport only if allowed by the
                // configuration info.
                if (hpx::util::get_entry_as<int>(cfg, "hpx.parcel.mpi.enable", "0"))
                {
                    return boost::make_shared<policies::mpi::connection_handler>(
                        cfg, on_start_thread, on_stop_thread);
                }
            }
#endif

            HPX_THROW_EXCEPTION(bad_parameter, "parcelport::create",
                "unsupported connection type 'connection_mpi'");
            break;

        default:
            HPX_THROW_EXCEPTION(bad_parameter, "parcelport::create",
                "unknown connection type");
            break;
        }

        return boost::shared_ptr<parcelport>();
    }

    ///////////////////////////////////////////////////////////////////////////
    parcelport::parcelport(util::runtime_configuration const& ini, locality const & here,
            std::string const& type)
      : parcels_(),
        here_(here),
        max_inbound_message_size_(ini.get_max_inbound_message_size()),
        max_outbound_message_size_(ini.get_max_outbound_message_size()),
        allow_array_optimizations_(true),
        allow_zero_copy_optimizations_(true),
        enable_security_(false),
        async_serialization_(false),
        enable_parcel_handling_(true)
    {
        std::string key("hpx.parcel.");
        key += type;

        if (hpx::util::get_entry_as<int>(ini, key + ".array_optimization", "1") == 0) {
            allow_array_optimizations_ = false;
            allow_zero_copy_optimizations_ = false;
        }
        else {
            if (hpx::util::get_entry_as<int>(ini, key + ".zero_copy_optimization", "1") == 0)
                allow_zero_copy_optimizations_ = false;
        }

        if(hpx::util::get_entry_as<int>(ini, key + ".enable_security", "0") != 0)
        {
            enable_security_ = true;
        }

        if(hpx::util::get_entry_as<int>(ini, key + ".async_serialization", "0") != 0)
        {
            async_serialization_ = true;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    boost::uint64_t get_max_inbound_size(parcelport& pp)
    {
        return pp.get_max_inbound_message_size();
    }
}}

