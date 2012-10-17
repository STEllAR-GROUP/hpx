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
#include <boost/foreach.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset
{
    ///////////////////////////////////////////////////////////////////////////
    void parcelport_connection::set_parcel(std::vector<parcel> const& pv)
    {
#if defined(HPX_DEBUG)
        BOOST_FOREACH(parcel const& p, pv)
        {
            naming::locality const locality_id = p.get_destination_locality();
            BOOST_ASSERT(locality_id == destination());
        }
#endif

        // reinitialize output wrapper
        output_data_.clear();

        // mark start of serialization
        util::high_resolution_timer timer;

        // guard against serialization errors
        try {
            // use the special io wrapper for serialization of the parcels
            output_data_.save(pv);
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

        // store the time spent on serialization and some other values
        send_data_.serialization_time_ = timer.elapsed_nanoseconds();
        send_data_.num_parcels_ = pv.size();
        send_data_.bytes_ = output_data_.bytes();
    }
}}

