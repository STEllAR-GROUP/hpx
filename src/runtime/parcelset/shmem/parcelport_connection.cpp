//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

#if defined(HPX_USE_SHMEM_PARCELPORT)
#include <hpx/runtime/parcelset/shmem/parcelport_connection.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/util/stringstream.hpp>
#include <hpx/traits/type_size.hpp>

#include <boost/iostreams/stream.hpp>
#include <boost/archive/basic_binary_oarchive.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

#include <stdexcept>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace shmem
{
    parcelport_connection::parcelport_connection(
            boost::asio::io_service& io_service,
            naming::locality const& here, naming::locality const& there,
            data_buffer_cache& cache,
            performance_counters::parcels::gatherer& parcels_sent,
            std::size_t connection_count)
      : window_(io_service), there_(there),
        parcels_sent_(parcels_sent), cache_(cache)
    {
        std::string fullname(here.get_address() + "." +
            boost::lexical_cast<std::string>(here.get_port()) + "." +
            boost::lexical_cast<std::string>(connection_count));

        window_.set_option(data_window::bound_to(fullname));
    }

    ///////////////////////////////////////////////////////////////////////////
    void parcelport_connection::set_parcel(std::vector<parcel> const& pv)
    {
#if defined(HPX_DEBUG)
        // make sure that all parcels require the same serialization filter and
        // that all of them go to the same locality
        util::binary_filter* filter = pv[0].get_serialization_filter();
        BOOST_FOREACH(parcel const& p, pv)
        {
            naming::locality const locality_id = p.get_destination_locality();
            BOOST_ASSERT(locality_id == destination());
            BOOST_ASSERT(filter == p.get_serialization_filter());
        }
#endif

        // collect argument sizes from parcels
        std::size_t arg_size = 0;

        // guard against serialization errors
        try {
            BOOST_FOREACH(parcel const & p, pv)
            {
                arg_size += traits::get_type_size(p);
            }

            // generate the name for this data_buffer
            std::string data_buffer_name(pv[0].get_parcel_id().to_string());

            // clear and preallocate out_buffer_ (or fetch from cache)
            out_buffer_ = get_data_buffer((arg_size * 12) / 10 + 1024,
                data_buffer_name);

            // mark start of serialization
            util::high_resolution_timer timer;

            {
                // Serialize the data
                util::portable_binary_oarchive archive(
                    out_buffer_.get_buffer(), 0, boost::archive::no_header);

                std::size_t count = pv.size();
                archive << count;

                BOOST_FOREACH(parcel const & p, pv)
                {
                    archive << p;
                }

                arg_size = archive.bytes_written();
            }

            // store the time required for serialization
            send_data_.serialization_time_ = timer.elapsed_nanoseconds();
        }
        catch (boost::archive::archive_exception const& e) {
            // We have to repackage all exceptions thrown by the
            // serialization library as otherwise we will loose the
            // e.what() description of the problem.
            HPX_THROW_EXCEPTION(serialization_error,
                "shmem::parcelport_connection::set_parcel",
                boost::str(boost::format(
                    "parcelport: parcel serialization failed, caught "
                    "boost::archive::archive_exception: %s") % e.what()));
            return;
        }
        catch (boost::system::system_error const& e) {
            HPX_THROW_EXCEPTION(serialization_error,
                "shmem::parcelport_connection::set_parcel",
                boost::str(boost::format(
                    "parcelport: parcel serialization failed, caught "
                    "boost::system::system_error: %d (%s)") %
                        e.code().value() % e.code().message()));
            return;
        }
        catch (std::exception const& e) {
            HPX_THROW_EXCEPTION(serialization_error,
                "shmem::parcelport_connection::set_parcel",
                boost::str(boost::format(
                    "parcelport: parcel serialization failed, caught "
                    "std::exception: %s") % e.what()));
            return;
        }

        send_data_.num_parcels_ = pv.size();
        send_data_.bytes_ = arg_size;
        send_data_.raw_bytes_ = out_buffer_.size();
    }
}}}

#endif
