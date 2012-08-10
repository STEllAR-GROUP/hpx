
//  Copyright (c) 2012 Julian Hornich
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>

#include <oclm/oclm.hpp>

int main()
{
    typedef std::vector< oclm::platform >::const_iterator    pIter;
    typedef std::vector< oclm::device >::const_iterator        dIter;

    std::vector< oclm::platform > plats( oclm::get_platforms() );;

    size_t nPlatforms(0);

    for ( pIter itPlatforms = plats.begin(); itPlatforms != plats.end(); ++itPlatforms )
    {
        std::cout << "Platform " << nPlatforms++ << ":\n";
        std::cout << "\tNAME:\t\t" << itPlatforms->get( oclm::platform_name ) << "\n";
        std::cout << "\tVENDOR:\t\t" << itPlatforms->get( oclm::platform_vendor ) << "\n";
        std::cout << "\tVERSION:\t" << itPlatforms->get( oclm::platform_version ) << "\n";
        std::cout << "\tPROFILE:\t" << itPlatforms->get( oclm::platform_profile ) << "\n";
        std::cout << "\tEXTENSIONS:\t" << itPlatforms->get( oclm::platform_extensions ) << "\n";

        size_t nDevices(0);

        const std::vector< oclm::device >& devs( itPlatforms->devices );

        for ( dIter itDevices = devs.begin(); itDevices != devs.end(); ++itDevices )
        {
            std::cout << "Device " << nDevices++ << ":\n";
            std::cout << "\tNAME:\t\t" << itDevices->get( oclm::device_name ) << "\n";
            std::cout << "\tVENDOR:\t\t" << itDevices->get( oclm::device_vendor ) << "\n";
            std::cout << "\tVERSION:\t" << itDevices->get( oclm::device_version ) << "\n";
            std::cout << "\tPROFILE:\t" << itDevices->get( oclm::device_profile ) << "\n";
            std::cout << "\tEXTENSIONS:\t" << itDevices->get( oclm::device_extensions ) << "\n";
        }

        std::cout << std::endl;
    }
}
