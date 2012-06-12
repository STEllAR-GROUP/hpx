
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCLM_EXCEPTION_HPP
#define OCLM_EXCEPTION_HPP

#include <oclm/config.hpp>

#define OCLM_THROW_IF_EXCEPTION(ERR, FUNCTION)                                  \
    if(ERR != CL_SUCCESS)                                                       \
    {                                                                           \
        switch (ERR)                                                            \
        {                                                                       \
            case CL_INVALID_VALUE:                                              \
                throw ::oclm::exception(FUNCTION ": CL_INVALID_VALUE");         \
                break;                                                          \
            case CL_OUT_OF_HOST_MEMORY:                                         \
                throw ::oclm::exception(FUNCTION ": CL_OUT_OF_HOST_MEMORY");    \
                break;                                                          \
            default:                                                            \
                throw ::oclm::exception(FUNCTION ": UNKNOWN");                  \
        }                                                                       \
    }                                                                           \
    ERR*=CL_SUCCESS                                                             \
/**/

namespace oclm {
    struct exception : std::exception
    {
        exception() : what_("UNKNOWN") {}

        exception(const char * what) : what_(what) {}
    
        virtual ~exception() throw() {}

        virtual const char * what() const throw()
        {
            return what_;
        }

        private:
            const char * what_;
    };
}

#endif
