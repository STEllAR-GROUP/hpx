
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCLM_HOST_HPP
#define OCLM_HOST_HPP

#include <CL/cl.h>

#include <vector>
#include <map>

namespace oclm {
    
    class host
    {
        public:
            host();

            static host & get();

        private:
            std::vector<cl_platform_id> platforms;
            std::map<cl_platform_id, std::vector<cl_device_id> > devices;
    };
}

#endif
