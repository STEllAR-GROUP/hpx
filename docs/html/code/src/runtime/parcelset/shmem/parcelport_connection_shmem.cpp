//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

#if defined(HPX_HAVE_PARCELPORT_SHMEM)
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
        parcels_sent_(parcels_sent), cache_(cache),
        archive_flags_(boost::archive::no_header)
    {
        std::string fullname(here.get_address() + "." +
            boost::lexical_cast<std::string>(here.get_port()) + "." +
            boost::lexical_cast<std::string>(connection_count));

        window_.set_option(data_window::bound_to(fullname));

        std::string array_optimization =
            get_config_entry("hpx.parcel.shmem.array_optimization", "1");

        if (boost::lexical_cast<int>(array_optimization) == 0)
            archive_flags_ |= util::disable_array_optimization;

        archive_flags_ |= util::disable_data_chunking;
    }

    ///////////////////////////////////////////////////////////////////////////
    void parcelport_connection::set_parcel(std::vector<parcel> const& pv)
    {
#if defined(HPX_DEBUG)
        // make sure that all parcels go to the same locality
        BOOST_FOREACH(parcel const& p, pv)
        {
            naming::locality const locality_id = p.get_destination_locality();
            HPX_ASSERT(locality_id == destination());
        }
#endif

        // collect argument sizes from parcels
        std::size_t arg_size = 0;
        boost::uint32_t dest_locality_id = pv[0].get_destination_locality_id();

        // guard against serialization errors
        try {
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
                        out_buffer_.get_buffer(), dest_locality_id, 0, archive_flags_);

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
            catch (hpx::exception const& e) {
                LPT_(fatal)
                    << "shmem::parcelport_connection::set_parcel: "
                       "caught hpx::exception: "
                    << e.what();
                hpx::report_error(boost::current_exception());
            }
            catch (boost::system::system_error const& e) {
                LPT_(fatal)
                    << "shmem::parcelport_connection::set_parcel: "
                       "caught boost::system::error: "
                    << e.what();
                hpx::report_error(boost::current_exception());
            }
            catch (boost::exception const&) {
                LPT_(fatal)
                    << "shmem::parcelport_connection::set_parcel: "
                       "caught boost::exception";
                hpx::report_error(boost::current_exception());
            }
            catch (std::exception const& e) {
                // We have to repackage all exceptions thrown by the
                // serialization library as otherwise we will loose the
                // e.what() description of the problem, due to slicing.
                boost::throw_exception(boost::enable_error_info(
                    hpx::exception(serialization_error, e.what())));
            }
        }
        catch (...) {
            LPT_(fatal)
                << "shmem::parcelport_connection::set_parcel: "
                   "caught unknown exception";
            hpx::report_error(boost::current_exception());
        }

        send_data_.num_parcels_ = pv.size();
        send_data_.bytes_ = arg_size;
        send_data_.raw_bytes_ = out_buffer_.size();
    }
}}}

#endif
