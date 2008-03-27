//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <string>

#include <boost/lexical_cast.hpp>

#include <hpx/distpx.hpp>

int main(int argc, char* argv[])
{
    // Check command line arguments.
    if (argc != 5)
    {
        std::cerr << "Usage: parcelset_client <px_ip> <px_port> <gas_ip> <gas_port>\n";
        std::cerr << "       try: parcelset_client localhost 7910 localhost 7912\n";
        return 1;
    }

    try {
        unsigned short px_port  = boost::lexical_cast<unsigned short>(argv[2]);
        unsigned short gas_port  = boost::lexical_cast<unsigned short>(argv[4]);

        // Start ParalleX services
        hpx::px_core px(argv[3], gas_port, argv[1], px_port, false);
        px.run(false);
        
        std::cout << "Parcelset (client) listening at port: " << px_port 
                  << std::flush << std::endl;
        
        // sleep for a second to give parcelset server a chance to startup
        boost::xtime xt;
        boost::xtime_get(&xt, boost::TIME_UTC);
        xt.sec += 1;
        boost::thread::sleep(xt);
                            
        parcelset::parcel p(1, new components::accumulator::init_action());
        parcelset::parcel_id id = px.get_parcelset().sync_put_parcel(p);

        std::cout << "Successfully sent parcel: " << std::hex << id << std::endl;

        px.stop();
    }
    catch (std::exception& e) {
        std::cerr << "std::exception caught: " << e.what() << "\n";
        return -1;
    }
    catch (...) {
        std::cerr << "unexpected exception caught\n";
        return -2;
    }
    return 0;
}

