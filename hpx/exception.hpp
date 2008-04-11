//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_EXCEPTION_MAR_24_2008_0929AM)
#define HPX_EXCEPTION_MAR_24_2008_0929AM

#include <exception>
#include <string>

#include <boost/assert.hpp>
#include <boost/system/error_code.hpp>

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
        version_unknown = 9,
        unknown_component_address = 10,
        duplicate_component_address = 11,
        invalid_status = 12,
        bad_parameter = 13,
        internal_server_error = 14,
        service_unavailable = 15,
        bad_request = 16,
        repeated_request = 17,
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
        "version_unknown",
        "unknown_component_address",
        "duplicate_component_address",
        "invalid status",
        "bad parameter",
        "internal_server_error",
        "service_unavailable",
        "bad_request",
        "repeated_request",
        ""
    };

    namespace detail 
    {
        class hpx_category : public boost::system::error_category
        {
        public:
            const char* name() const
            {
                return "HPX";
            }

            std::string message(int value) const
            {
                switch (value) {
                case success:
                case no_success:
                case not_implemented:
                case out_of_memory:
                case bad_action_code:
                case bad_component_type:
                case network_error:
                case version_too_new:
                case version_too_old:
                case unknown_component_address:
                case duplicate_component_address:
                case invalid_status:
                case bad_parameter:
                case repeated_request:
                    return std::string("HPX(") + error_names[value] + ")";

                default:
                    break;
                }
                return "HPX(Unknown error)";
            }
        };

    } // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    //  Define the HPX error category
    inline boost::system::error_category const& get_hpx_category()
    {
        static detail::hpx_category instance;
        return instance;
    }

    static const boost::system::error_category& hpx_category =
        hpx::get_hpx_category();

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
    
    ///////////////////////////////////////////////////////////////////////////
    inline boost::system::error_code make_error_code(error e)
    {
        return boost::system::error_code(
            static_cast<int>(e), get_hpx_category());
    }

///////////////////////////////////////////////////////////////////////////////
}

///////////////////////////////////////////////////////////////////////////////
namespace boost { namespace system
{
    // make sure our errors get recognized by the Boost.System library
    template<> struct is_error_code_enum<hpx::error>
    {
        static const bool value = true;
    };
}}

#endif



