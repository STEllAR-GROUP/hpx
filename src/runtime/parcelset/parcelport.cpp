//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This is needed to make everything work with the Intel MPI library header
#include <hpx/config/defines.hpp>
#if defined(HPX_HAVE_PARCELPORT_MPI)
#include <mpi.h>
#endif

#include <hpx/hpx_fwd.hpp>
#if defined(HPX_HAVE_PARCELPORT_TCPIP)
#include <hpx/runtime/parcelset/tcp/parcelport.hpp>
#endif
#if defined(HPX_HAVE_PARCELPORT_SHMEM)
#  include <hpx/runtime/parcelset/shmem/parcelport.hpp>
#endif
#if defined(HPX_HAVE_PARCELPORT_IBVERBS)
#  include <hpx/runtime/parcelset/ibverbs/parcelport.hpp>
#endif
#if defined(HPX_HAVE_PARCELPORT_MPI)
#  include <hpx/runtime/parcelset/mpi/parcelport.hpp>
#endif
#include <hpx/util/io_service_pool.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/exception.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

namespace hpx { namespace parcelset
{
    boost::shared_ptr<parcelport> parcelport::create_bootstrap(
        util::runtime_configuration const& cfg,
        HPX_STD_FUNCTION<void(std::size_t, char const*)> const& on_start_thread,
        HPX_STD_FUNCTION<void()> const& on_stop_thread)
    {
        std::string pptype = cfg.get_entry("hpx.parcel.bootstrap", "tcpip");

        int type = get_connection_type_from_name(pptype);
        if (type == connection_unknown)
            type = connection_tcpip;

        return create(type, cfg, on_start_thread, on_stop_thread);
    }

    boost::shared_ptr<parcelport> parcelport::create(int type,
        util::runtime_configuration const& cfg,
        HPX_STD_FUNCTION<void(std::size_t, char const*)> const& on_start_thread,
        HPX_STD_FUNCTION<void()> const& on_stop_thread)
    {
        switch(type) {
        case connection_tcpip:
            {
#if defined(HPX_HAVE_PARCELPORT_TCPIP)
                std::string enable_tcpip =
                    cfg.get_entry("hpx.parcel.tcpip.enable", "1");

                if (boost::lexical_cast<int>(enable_tcpip))
                {
                    return boost::make_shared<parcelset::tcp::parcelport>(
                        cfg, on_start_thread, on_stop_thread);
                }
#endif

                HPX_THROW_EXCEPTION(bad_parameter, "parcelport::create",
                    "unsupported connection type 'connection_tcpip'");
            }

        case connection_shmem:
            {
#if defined(HPX_HAVE_PARCELPORT_SHMEM)
                // Create shmem based parcelport only if allowed by the
                // configuration info.
                std::string enable_shmem =
                    cfg.get_entry("hpx.parcel.shmem.enable", "0");

                if (boost::lexical_cast<int>(enable_shmem))
                {
                    return boost::make_shared<parcelset::shmem::parcelport>(
                        cfg, on_start_thread, on_stop_thread);
                }
#endif
                HPX_THROW_EXCEPTION(bad_parameter, "parcelport::create",
                    "unsupported connection type 'connection_shmem'");
            }
            break;

        case connection_ibverbs:
#if defined(HPX_HAVE_PARCELPORT_IBVERBS)
            {
                // Create ibverbs based parcelport only if allowed by the
                // configuration info.
                std::string enable_ibverbs =
                    cfg.get_entry("hpx.parcel.ibverbs.enable", "0");

                if (boost::lexical_cast<int>(enable_ibverbs))
                {
                    return boost::make_shared<parcelset::ibverbs::parcelport>(
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
#if defined(HPX_HAVE_PARCELPORT_MPI)
            {
                // Create MPI based parcelport only if allowed by the
                // configuration info.
                std::string enable_mpi =
                    cfg.get_entry("hpx.parcel.mpi.enable", "0");

                if (boost::lexical_cast<int>(enable_mpi))
                {
                    return boost::make_shared<parcelset::mpi::parcelport>(
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
    parcelport::parcelport(util::runtime_configuration const& ini)
      : parcels_(),
        here_(ini.get_parcelport_address()),
        max_message_size_(ini.get_max_message_size()),
        allow_array_optimizations_(true),
        allow_zero_copy_optimizations_(true)
    {
        std::string array_optimization =
            ini.get_entry("hpx.parcel.array_optimization", "1");
        if (boost::lexical_cast<int>(array_optimization) == 0)
            allow_array_optimizations_ = false;

        std::string zero_copy_optimization =
            ini.get_entry("hpx.parcel.zero_copy_optimization", array_optimization);
        if (boost::lexical_cast<int>(zero_copy_optimization) == 0)
            allow_zero_copy_optimizations_ = false;
    }

    ///////////////////////////////////////////////////////////////////////////
    // default implementation, just forward to single parcel version
    void parcelport::put_parcels(std::vector<parcel> const & parcels,
            std::vector<write_handler_type> const& handlers)
    {
        if (parcels.size() != handlers.size())
        {
            HPX_THROW_EXCEPTION(bad_parameter, "parcelport::put_parcels",
                "mismatched number of parcels and handlers");
            return;
        }

        for (std::size_t i = 0; i != parcels.size(); ++i)
        {
            put_parcel(parcels[i], handlers[i]);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    void parcelport::report_potential_connection_error(
        naming::locality const& locality_id, naming::gid_type const& parcel_id,
        error_code const& ec)
    {
        typedef pending_parcels_map::iterator iterator;

        // If there was an error, we might be safe if there are no parcels
        // to be sent anymore (some other thread already picked them up)
        // or if there are parcels, but the parcel we were about to sent
        // has been already processed.
        lcos::local::spinlock::scoped_lock l(mtx_);

        iterator it = pending_parcels_.find(locality_id);
        if (it != pending_parcels_.end())
        {
            map_second_type& data = it->second;

            std::vector<parcel>::iterator end = data.first.end();
            std::vector<write_handler_type>::iterator fit = data.second.begin();
            for (std::vector<parcel>::iterator pit = data.first.begin();
                    pit != end; ++pit, ++fit)
            {
                if ((*pit).get_parcel_id() == parcel_id)
                {
                    // our parcel is still here, bailing out
                    throw hpx::detail::access_exception(ec);
                }
            }
        }
    }
}}

