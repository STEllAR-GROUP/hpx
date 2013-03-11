//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/parcelset/tcp/parcelport.hpp>
#if defined(HPX_HAVE_PARCELPORT_SHMEM)
#  include <hpx/runtime/parcelset/shmem/parcelport.hpp>
#endif
#if defined(HPX_HAVE_PARCELPORT_IBVERBS)
#  include <hpx/runtime/parcelset/ibverbs/parcelport.hpp>
#endif
#include <hpx/util/io_service_pool.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/exception.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

namespace hpx { namespace parcelset
{
    boost::shared_ptr<parcelport> parcelport::create(connection_type type,
        util::runtime_configuration const& cfg,
        HPX_STD_FUNCTION<void(std::size_t, char const*)> const& on_start_thread,
        HPX_STD_FUNCTION<void()> const& on_stop_thread)
    {
        switch(type) {
        case connection_tcpip:
            return boost::make_shared<parcelset::tcp::parcelport>(
                cfg, on_start_thread, on_stop_thread);

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

                if(boost::lexical_cast<int>(enable_ibverbs))
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

        default:
            HPX_THROW_EXCEPTION(bad_parameter, "parcelport::create",
                "unknown connection type");
            break;
        }

        return boost::shared_ptr<parcelport>();
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
        util::spinlock::scoped_lock l(mtx_);

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

