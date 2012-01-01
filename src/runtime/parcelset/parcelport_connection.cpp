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

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset
{
    ///////////////////////////////////////////////////////////////////////////
    void parcelport_connection::set_parcel (parcel const& p)
    {
        typedef util::container_device<std::vector<char> > io_device_type;

        // guard against serialization errors
        try {
            // create a special io stream on top of out_buffer_
            out_buffer_.clear();
            boost::iostreams::stream<io_device_type> io(out_buffer_);

            // Serialize the data
#if HPX_USE_PORTABLE_ARCHIVES != 0
            util::portable_binary_oarchive archive(io);
#else
            boost::archive::binary_oarchive archive(io);
#endif
            archive << p;
        }
        catch (std::exception const& e) {
            hpx::util::osstream strm;
            strm << "parcelport: parcel serialization failed: " << e.what()
                 << " parcel(" << p << ")";
            HPX_THROW_EXCEPTION(no_success,
                "parcelport_connection::set_parcel",
                hpx::util::osstream_get_string(strm));
            return;
        }

        out_priority_ = boost::integer::ulittle8_t(p.get_thread_priority());
        out_size_ = out_buffer_.size();
    }
}}

