//  Copyright (c) 2007-2008 Hartmut Kaiser
//  Copyright (c) 2007 Richard D. Guidry Jr.
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_NAMING_LOCALITY_MAR_24_2008_0942AM)
#define HPX_NAMING_LOCALITY_MAR_24_2008_0942AM

#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <boost/cstdint.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/serialization.hpp>

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/asio_util.hpp>

///////////////////////////////////////////////////////////////////////////////
///  version of GAS reply structure
#define HPX_LOCALITY_VERSION   0x20

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming
{
    ///////////////////////////////////////////////////////////////////////////
    class locality
    {
    public:
        locality() 
        {}
        
        locality(std::string const& addr, unsigned short port) 
        {
            try {
                if (!util::get_endpoint(addr, port, endpoint_)) {
                    using boost::asio::ip::tcp;
                    
                    // resolve the given address
                    boost::asio::io_service io_service;
                    tcp::resolver resolver(io_service);
                    tcp::resolver::query query(
                        !addr.empty() ? addr : boost::asio::ip::host_name(), 
                        boost::lexical_cast<std::string>(port));

                    endpoint_ = *resolver.resolve(query);
                }
            }
            catch (boost::system::error_code const& e) {
                throw hpx::exception(network_error, e.message());
            }
        }
        
        locality(boost::asio::ip::address addr, unsigned short port) 
          : endpoint_(addr, port) 
        {
        }
        
        locality(boost::asio::ip::tcp::endpoint ep) 
          : endpoint_(ep) 
        {
        }
        
        locality& operator= (boost::asio::ip::tcp::endpoint ep)
        {
            endpoint_ = ep;
            return *this;
        }
        
        /// access the stored IP address
        boost::asio::ip::tcp::endpoint const& get_endpoint() const 
        {
            return endpoint_;
        }
        
        friend bool operator!=(locality const& lhs, locality const& rhs)
        {
            return lhs.endpoint_ != rhs.endpoint_;
        }

        friend bool operator==(locality const& lhs, locality const& rhs)
        {
            return lhs.endpoint_ == rhs.endpoint_;
        }

        friend bool operator< (locality const& lhs, locality const& rhs)
        {
            return lhs.endpoint_ < rhs.endpoint_;
        }
        
        friend bool operator> (locality const& lhs, locality const& rhs)
        {
            return !(lhs.endpoint_ < rhs.endpoint_) && lhs.endpoint_ != rhs.endpoint_;
        }
        
    private:
        friend std::ostream& operator<< (std::ostream& os, locality const& l);
        
        // serialization support    
        friend class boost::serialization::access;
    
        template<class Archive>
        void save(Archive & ar, const unsigned int version) const
        {
        boost::uint16_t port = endpoint_.port();    
            
            if (endpoint_.address().is_v4()) {
            boost::uint32_t ip = endpoint_.address().to_v4().to_ulong();
            bool is_v4 = true;
            
                ar << is_v4 << ip << port;
            }
            else {
            boost::asio::ip::address_v6 addr = endpoint_.address().to_v6();
            
                if (addr.is_v4_mapped() || addr.is_v4_compatible()) {
                boost::uint32_t ip = addr.to_v4().to_ulong();
                bool is_v4 = true;
                
                    ar << is_v4 << ip << port;
                }
                else {
                std::string bytes (addr.to_string());
                bool is_v4 = false;
                
                    ar << is_v4 << bytes << port;
                }
            }
        }

        template<class Archive>
        void load(Archive & ar, const unsigned int version)
        {
            if (version > HPX_LOCALITY_VERSION) {
                throw exception(version_too_new, 
                    "trying to load locality with unknown version");
            }
            
        bool is_v4 = false;
        boost::uint16_t port = 0;    
            
            ar >> is_v4;
            if (is_v4) {
            boost::uint32_t ip = 0;

                ar >> ip;
                ar >> port;
            
                endpoint_.address(boost::asio::ip::address_v4(ip));
                endpoint_.port(port);
            }
            else {
            std::string bytes;
            
                ar >> bytes;
                ar >> port;
                endpoint_.address(boost::asio::ip::address_v6::from_string(bytes));
                endpoint_.port(port);
            }
        }
        BOOST_SERIALIZATION_SPLIT_MEMBER()

    private:
        boost::asio::ip::tcp::endpoint endpoint_;
    };

    inline std::ostream& operator<< (std::ostream& os, locality const& l)
    {
        os << std::dec << l.endpoint_;
        return os;
    }
    
///////////////////////////////////////////////////////////////////////////////
}}

///////////////////////////////////////////////////////////////////////////////
// this is the current version of the parcel serialization format
// this definition needs to be in the global namespace
BOOST_CLASS_VERSION(hpx::naming::locality, HPX_LOCALITY_VERSION)
BOOST_CLASS_TRACKING(hpx::naming::locality, boost::serialization::track_never)

#endif

