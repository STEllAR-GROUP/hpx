
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <oclm/host.hpp>
#include <oclm/util/static.hpp>
#include <oclm/exception.hpp>

#include <boost/move/move.hpp>

namespace {
    std::vector<cl_platform_id> acquire_platforms()
    {
        std::vector<cl_platform_id> platforms;
        /*
        cl_uint n = 0;
        cl_int err = ::clGetPlatformIDs(0, NULL, &n);
        OCLM_THROW_IF_EXCEPTION(err, "clGetPlatformIDs");

        std::vector<cl_platform_id> platforms(n);

        err = ::clGetPlatformIDs(n, &platforms[0], NULL);
        OCLM_THROW_IF_EXCEPTION(err, "clGetPlatformIDs");
        */

        return platforms;
    }

    std::map<cl_platform_id, std::vector<cl_device_id> >
        acquire_devices(std::vector<cl_platform_id> const & platforms)
    {
        std::map<cl_platform_id, std::vector<cl_device_id> > devs;
        /*
        typedef std::vector<cl_platform_id>::const_iterator platform_iterator;


        for(platform_iterator it = platforms.begin(); it != platforms.end(); ++it)
        {
            cl_uint n = 0;
            cl_int err = ::clGetDeviceIDs(*it, CL_DEVICE_TYPE_ALL, 0, NULL, &n);
            OCLM_THROW_IF_EXCEPTION(err, "clGetDeviceIDs");
            
            std::vector<cl_device_id> devices(n);
            err = ::clGetDeviceIDs(*it, CL_DEVICE_TYPE_ALL, n, &devices[0], NULL);
            OCLM_THROW_IF_EXCEPTION(err, "clGetDeviceIDs");

            devs.insert(std::make_pair(*it, boost::move(devices)));
        }

        */
        return devs;
    }
}

namespace oclm
{
    host::host()
        : platforms(::acquire_platforms())
        , devices(::acquire_devices(platforms))
    {
    }

    host & host::get()
    {
        util::static_<host> instance;

        return instance.get();
    }
}
