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
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ip/udp.hpp>
#include <boost/asio/ip/icmp.hpp>

#include <hpx/util/container_device.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/util/serialize_asio_basic_endpoint.hpp>

int main()
{
    typedef std::vector<char> buffer_type;
    typedef hpx::util::container_device<buffer_type> device_type;
    typedef boost::iostreams::stream<device_type> stream_type;

    #if HPX_USE_PORTABLE_ARCHIVES != 0
        typedef hpx::util::portable_binary_oarchive oarchive_type;
        typedef hpx::util::portable_binary_iarchive iarchive_type;
    #else
        typedef boost::archive::binary_oarchive oarchive_type;
        typedef boost::archive::binary_iarchive iarchive_type;
    #endif
    
    using boost::asio::ip::address;

    { // icmp
        typedef boost::asio::ip::icmp::endpoint endpoint;
    
        buffer_type buffer; 
        endpoint data(address::from_string("192.168.2.9"), 80);
    
        { // save 
            stream_type stream(buffer);
            oarchive_type archive(stream);
            archive & data;
        }
    
        { // load
            stream_type stream(buffer);
            iarchive_type archive(stream);
            endpoint loaded;
            archive & loaded;
    
            HPX_TEST_EQ(data, loaded);
        }
    }

    { // tcp
        typedef boost::asio::ip::tcp::endpoint endpoint;
    
        buffer_type buffer; 
        endpoint data(address::from_string("192.168.2.9"), 80);
    
        { // save 
            stream_type stream(buffer);
            oarchive_type archive(stream);
            archive & data;
        }
    
        { // load
            stream_type stream(buffer);
            iarchive_type archive(stream);
            endpoint loaded;
            archive & loaded;
    
            HPX_TEST_EQ(data, loaded);
        }
    }

    { // udp
        typedef boost::asio::ip::udp::endpoint endpoint;
    
        buffer_type buffer; 
        endpoint data(address::from_string("192.168.2.9"), 80);
    
        { // save 
            stream_type stream(buffer);
            oarchive_type archive(stream);
            archive & data;
        }
    
        { // load
            stream_type stream(buffer);
            iarchive_type archive(stream);
            endpoint loaded;
            archive & loaded;
    
            HPX_TEST_EQ(data, loaded);
        }
    }

    return boost::report_errors();
}

