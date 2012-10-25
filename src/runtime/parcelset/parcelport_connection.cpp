//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <stdexcept>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/parcelset/parcelport_connection.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
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
                arg_size += p.get_action()->get_argument_size();
                priority = (std::max)(p.get_thread_priority(), priority);
            }

            out_buffer_.reserve(arg_size*2);

            // mark start of serialization
            util::high_resolution_timer timer;

            {
                // Serialize the data
                util::portable_binary_oarchive archive(out_buffer_,
                    boost::archive::no_header);

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
        send_data_.argument_bytes_ = arg_size;
    }
}}

