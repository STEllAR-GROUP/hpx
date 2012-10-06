//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <stdexcept>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/parcelset/parcelport_connection.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/util/container_device.hpp>
#include <hpx/util/stringstream.hpp>

#include <boost/iostreams/stream.hpp>
#include <boost/archive/basic_binary_oarchive.hpp>
#include <boost/format.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset
{
    ///////////////////////////////////////////////////////////////////////////
    void parcelport_connection::set_parcel(std::vector<parcel> const& pv)
    {
        // we choose the highest priority of all parcels for this message
        threads::thread_priority priority = threads::thread_priority_default;

#if defined(HPX_DEBUG)
        BOOST_FOREACH(parcel const& p, pv)
        {
            naming::locality const locality_id = p.get_destination_locality();
            BOOST_ASSERT(locality_id == destination());
        }
#endif

        // guard against serialization errors
        try {
            // create a special io stream on top of out_buffer_
            out_buffer_.clear();

            // mark start of serialization
            send_data_.serialization_time_ = timer_.elapsed_nanoseconds();

            {
                typedef util::container_device<std::vector<char> > io_device_type;
                boost::iostreams::stream<io_device_type> io(out_buffer_);

                // Serialize the data
                util::portable_binary_oarchive archive(io);

                std::size_t count = pv.size();
                archive << count;

                BOOST_FOREACH(parcel const & p, pv)
                {
                    priority = (std::max)(p.get_thread_priority(), priority);
                    archive << p;
                }
            }

            // store the time required for serialization
            send_data_.serialization_time_ =
                timer_.elapsed_nanoseconds() - send_data_.serialization_time_;
        }
        catch (boost::archive::archive_exception const& e) {
            // We have to repackage all exceptions thrown by the
            // serialization library as otherwise we will loose the
            // e.what() description of the problem.
            HPX_RETHROW_EXCEPTION(serialization_error,
                "parcelport_connection::set_parcel",
                boost::str(boost::format(
                    "parcelport: parcel serialization failed, caught "
                    "boost::archive::archive_exception: %s") % e.what()));
            return;
        }
        catch (boost::system::system_error const& e) {
            HPX_RETHROW_EXCEPTION(serialization_error,
                "parcelport_connection::set_parcel",
                boost::str(boost::format(
                    "parcelport: parcel serialization failed, caught "
                    "boost::system::system_error: %d (%s)") %
                        e.code().value() % e.code().message()));
            return;
        }
        catch (std::exception const& e) {
            HPX_RETHROW_EXCEPTION(serialization_error,
                "parcelport_connection::set_parcel",
                boost::str(boost::format(
                    "parcelport: parcel serialization failed, caught "
                    "std::exception: %s") % e.what()));
            return;
        }

        out_priority_ = boost::integer::ulittle8_t(priority);
        out_size_ = out_buffer_.size();

        send_data_.num_parcels_ = pv.size();
        send_data_.bytes_ = out_buffer_.size();
    }
}}

