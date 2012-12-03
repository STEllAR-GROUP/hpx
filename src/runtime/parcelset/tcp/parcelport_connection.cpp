//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <stdexcept>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/parcelset/tcp/parcelport_connection.hpp>
#include <hpx/runtime/parcelset/parcelport.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/util/stringstream.hpp>
#include <hpx/traits/type_size.hpp>

#include <boost/iostreams/stream.hpp>
#include <boost/archive/basic_binary_oarchive.hpp>
#include <boost/format.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace tcp
{
    parcelport_connection::parcelport_connection(boost::asio::io_service& io_service,
            naming::locality const& locality_id,
            util::connection_cache<parcelport_connection, naming::locality>& cache,
            performance_counters::parcels::gatherer& parcels_sent)
      : socket_(io_service), out_priority_(0), out_size_(0), there_(locality_id),
        connection_cache_(cache), parcels_sent_(parcels_sent), 
        archive_flags_(boost::archive::no_header)
    {
#ifdef BOOST_BIG_ENDIAN
        std::string endian_out = get_config_entry("hpx.parcel.endian_out", "big");
#else
        std::string endian_out = get_config_entry("hpx.parcel.endian_out", "little");
#endif
        if (endian_out == "little")
            archive_flags_ |= util::endian_little;
        else if (endian_out == "big")
            archive_flags_ |= util::endian_big;
        else {
            BOOST_ASSERT(endian_out =="little" || endian_out == "big");
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    void parcelport_connection::set_parcel(std::vector<parcel> const& pv)
    {
        // we choose the highest priority of all parcels for this message
        threads::thread_priority priority = threads::thread_priority_default;

        // collect argument sizes from parcels
        std::size_t arg_size = 0;

#if defined(HPX_DEBUG)
        BOOST_FOREACH(parcel const& p, pv)
        {
            naming::locality const locality_id = p.get_destination_locality();
            BOOST_ASSERT(locality_id == destination());
        }
#endif

        // guard against serialization errors
        try {
            // clear and preallocate out_buffer_
            out_buffer_.clear();

            BOOST_FOREACH(parcel const & p, pv)
            {
                arg_size += traits::get_type_size(p);
                priority = (std::max)(p.get_thread_priority(), priority);
            }

            out_buffer_.reserve(arg_size*2);

            // mark start of serialization
            util::high_resolution_timer timer;

            {
                // Serialize the data
                util::portable_binary_oarchive archive(out_buffer_,
                    archive_flags_);

                std::size_t count = pv.size();
                archive << count;

                BOOST_FOREACH(parcel const & p, pv)
                {
                    archive << p;
                }
            }

            // store the time required for serialization
            send_data_.serialization_time_ = timer.elapsed_nanoseconds();
        }
        catch (boost::archive::archive_exception const& e) {
            // We have to repackage all exceptions thrown by the
            // serialization library as otherwise we will loose the
            // e.what() description of the problem.
            HPX_RETHROW_EXCEPTION(serialization_error,
                "tcp::parcelport_connection::set_parcel",
                boost::str(boost::format(
                    "parcelport: parcel serialization failed, caught "
                    "boost::archive::archive_exception: %s") % e.what()));
            return;
        }
        catch (boost::system::system_error const& e) {
            HPX_RETHROW_EXCEPTION(serialization_error,
                "tcp::parcelport_connection::set_parcel",
                boost::str(boost::format(
                    "parcelport: parcel serialization failed, caught "
                    "boost::system::system_error: %d (%s)") %
                        e.code().value() % e.code().message()));
            return;
        }
        catch (std::exception const& e) {
            HPX_RETHROW_EXCEPTION(serialization_error,
                "tcp::parcelport_connection::set_parcel",
                boost::str(boost::format(
                    "parcelport: parcel serialization failed, caught "
                    "std::exception: %s") % e.what()));
            return;
        }

        out_priority_ = boost::integer::ulittle8_t(priority);
        out_size_ = out_buffer_.size();

        send_data_.num_parcels_ = pv.size();
        send_data_.bytes_ = out_buffer_.size();
        send_data_.type_bytes_ = arg_size;
    }
}}}

