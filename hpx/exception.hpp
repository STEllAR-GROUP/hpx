//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_EXCEPTION_MAR_24_2008_0929AM)
#define HPX_EXCEPTION_MAR_24_2008_0929AM

#include <exception>
#include <string>

#include <boost/assert.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx  
{
    ///////////////////////////////////////////////////////////////////////////
    enum error
    {
        success = 0,
        no_success = 1,
        not_implemented = 2,
        out_of_memory = 3,
        bad_action_code = 4,
        bad_component_type = 5,
        network_error = 6,
        version_too_new = 7,
        version_too_old = 8,
        unknown_component_address = 9,
        duplicate_component_address = 10,
        invalid_status = 11,
        last_error
    };
    
    char const* const error_names[] = 
    {
        "success",
        "no_success",
        "not_implemented",
        "out_of_memory",
        "bad_action_code",
        "bad_component_type",
        "network_error",
        "version_too_new",
        "version_too_old",
        "unknown_component_address",
        "duplicate_component_address",
        "invalid status",
        ""
    };

    ///////////////////////////////////////////////////////////////////////////
    class exception : public std::exception
    {
    public:
        exception(error e, char const* msg) 
          : error_code_(e) 
        {
            BOOST_ASSERT(e >= success && e < last_error);
            msg_ = std::string("HPX(") + error_names[e] + "): " + msg;
        }
        exception(error e, std::string msg) 
          : error_code_(e) 
        {
            BOOST_ASSERT(e >= success && e < last_error);
            msg_ = std::string("HPX(") + error_names[e] + "): " + msg;
        }
        ~exception (void) throw() 
        {
        }
        
        const char* what() const throw() 
        { 
            return msg_.c_str();
        }

        error get_error() const throw() 
        { 
            return error_code_; 
        }

    private:
        std::string msg_;
        error error_code_;
    };
    
///////////////////////////////////////////////////////////////////////////////
}

#endif



