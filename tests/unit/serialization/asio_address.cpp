////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <vector>

#include <boost/iostreams/stream.hpp>
#include <boost/archive/basic_binary_iarchive.hpp>
#include <boost/archive/basic_binary_oarchive.hpp>

#include <hpx/util/container_device.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/util/serialize_asio_address.hpp>

int main()
{
    using boost::asio::ip::address;

    typedef std::vector<char> buffer_type;
    typedef hpx::util::container_device<buffer_type> device_type;
    typedef boost::iostreams::stream<device_type> stream_type;
    typedef address data_type;
    
    #if HPX_USE_PORTABLE_ARCHIVES != 0
        typedef hpx::util::portable_binary_oarchive oarchive_type;
        typedef hpx::util::portable_binary_iarchive iarchive_type;
    #else
        typedef boost::archive::binary_oarchive oarchive_type;
        typedef boost::archive::binary_iarchive iarchive_type;
    #endif

    { // ipv4 
        buffer_type buffer; 
        data_type data = address::from_string("192.0.2.235");
   
        HPX_SANITY(data.is_v4());
 
        { // save 
            stream_type stream(buffer);
            oarchive_type archive(stream);
            archive & data;
        
            HPX_SANITY(data.is_v4());
        }
    
        { // load
            stream_type stream(buffer);
            iarchive_type archive(stream);
            data_type loaded;
            archive & loaded;

            HPX_SANITY(data.is_v4()); 
            HPX_TEST(loaded.is_v4()); 
            HPX_TEST_EQ(data, loaded);
        }
    }

    { // ipv6
        buffer_type buffer; 
        data_type data = address::from_string("2001:0DB8:AC10:FE01::");
    
        { // save 
            stream_type stream(buffer);
            oarchive_type archive(stream);
            archive & data;
            
            HPX_SANITY(data.is_v6()); 
        }
    
        { // load
            stream_type stream(buffer);
            iarchive_type archive(stream);
            data_type loaded;
            archive & loaded;
    
            HPX_SANITY(data.is_v6()); 
            HPX_TEST(loaded.is_v6()); 
            HPX_TEST_EQ(data, loaded);
        }
    }

    return hpx::util::report_errors();
}

