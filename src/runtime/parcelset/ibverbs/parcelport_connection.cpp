//  Copyright (c)      2013 Thomas Heller
//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

#if defined(HPX_HAVE_PARCELPORT_IBVERBS)
#include <hpx/runtime/parcelset/ibverbs/parcelport_connection.hpp>
#include <hpx/runtime/parcelset/parcelport.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/util/stringstream.hpp>
#include <hpx/traits/type_size.hpp>

#include <boost/iostreams/stream.hpp>
#include <boost/archive/basic_binary_oarchive.hpp>
#include <boost/format.hpp>

#include <unistd.h>
#include <sys/param.h>
#if defined(__FreeBSD__)
#define EXEC_PAGESIZE PAGE_SIZE
#endif

#include <stdexcept>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace ibverbs
{
    parcelport_connection::parcelport_connection(
            boost::asio::io_service& io_service,
            naming::locality const& there,
            performance_counters::parcels::gatherer& parcels_sent)
      : context_(io_service), out_priority_(0), out_size_(0), out_data_size_(0),
        there_(there), parcels_sent_(parcels_sent),
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
        boost::system::error_code ec;
        std::string buffer_size_str = get_config_entry("hpx.parcel.ibverbs.buffer_size", "4096");
        context_.set_buffer_size(boost::lexical_cast<std::size_t>(buffer_size_str), ec);

        std::string sync_threshold_str = get_config_entry("hpx.parcel.ibverbs.sync_threshold", "0");
        sync_threshold_ = boost::lexical_cast<std::size_t>(sync_threshold_str);
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
        // we choose the highest priority of all parcels for this message
        threads::thread_priority priority = threads::thread_priority_default;

        // collect argument sizes from parcels
        std::size_t arg_size = 0;

        // guard against serialization errors
        try {
            out_buffer_.clear();
            BOOST_FOREACH(parcel const & p, pv)
            {
                arg_size += traits::get_type_size(p);
            }

            out_buffer_.reserve(arg_size*2);

            // mark start of serialization
            util::high_resolution_timer timer;

            {
                // Serialize the data
                // Serialize the data
                util::binary_filter* filter = pv[0].get_serialization_filter();
                int archive_flags = archive_flags_;
                if (filter) {
                    filter->set_max_compression_length(out_buffer_.capacity());
                    archive_flags |= util::enable_compression;
                }

                util::portable_binary_oarchive archive(
                    out_buffer_, filter, archive_flags);

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

        out_priority_ = boost::integer::ulittle8_t(priority);
        out_size_ = out_buffer_.size();
        out_data_size_ = arg_size; 

        send_data_.num_parcels_ = pv.size();
        send_data_.bytes_ = arg_size;
        send_data_.raw_bytes_ = out_buffer_.size();
    }
}}}

#endif
