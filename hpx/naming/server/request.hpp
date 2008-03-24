//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_NAMING_SERVER_REQUEST_MAR_24_2008_0941AM)
#define HPX_NAMING_SERVER_REQUEST_MAR_24_2008_0941AM

#include <boost/cstdint.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/serialization.hpp>

#include <hpx/naming/name.hpp>
#include <hpx/naming/locality.hpp>

///////////////////////////////////////////////////////////////////////////////
///  version of GAS request structure
#define HPX_REQUEST_VERSION   0x20

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming { namespace server 
{
    /// the commands supported by the name server
    enum name_server_command
    {
        command_unknown = 0,
        command_getprefix = 1,      /// return a unique prefix for the requesting site
        command_bind = 2,           /// bind an address to a global id
        command_unbind = 3,         /// remove binding for a global id
        command_resolve = 4,        /// resolve a global id to an address
        command_queryid = 5,        /// query for a global id associated with a namespace name (string)
        command_registerid = 6,     /// associate a namespace name with a global id
        command_unregisterid = 7,   /// remove association of a namespace name with a global id
        command_statistics = 8,     /// return some usage statistics
    };

    /// A request received from a client.
    class request
    {
    public:
        request(name_server_command c = command_unknown) 
          : command_(c), id_(0)
        {}
        
        request(name_server_command c, locality l) 
          : command_(c), id_(0), site_(l)
        {}
        
        request(name_server_command c, naming::id_type id) 
          : command_(c), id_(id)
        {}

        request(name_server_command c, std::string const& ns_name, naming::id_type id) 
          : command_(c), id_(id), ns_name_(ns_name)
        {}

        request(name_server_command c, std::string const& ns_name) 
          : command_(c), id_(0), ns_name_(ns_name)
        {}

        request(name_server_command c, boost::uint64_t id, address const& addr) 
          : command_(c), id_(id), addr_(addr)
        {}

        boost::uint8_t get_command() const
        {
            return command_;
        }
        
        naming::id_type get_id() const
        {
            return id_;
        }
        
        naming::locality get_site() const
        {
            return site_;
        }
        
        naming::address get_address() const 
        {
            return addr_;
        }
        
        std::string get_name() const
        {
            return ns_name_;
        }
        
    private:
        // serialization support    
        friend class boost::serialization::access;
    
        template<class Archive>
        void save(Archive & ar, const unsigned int version) const
        {
            ar << command_;
            ar << site_;
            if (command_ == command_resolve || command_ == command_unbind) {
                ar << id_.id_;
            }
            else if (command_ == command_bind) {
                ar << id_.id_;
                ar << addr_.locality_;
                ar << addr_.type_;
                ar << addr_.address_;
            }
            else if (command_ == command_queryid || 
                     command_ == command_unregisterid) 
            {
                ar << ns_name_;
            }
            else if (command_ == command_registerid) {
                ar << id_.id_;
                ar << ns_name_;
            }
        }

        template<class Archive>
        void load(Archive & ar, const unsigned int version)
        {
            if (version > HPX_REQUEST_VERSION) {
                throw exception(version_too_new, 
                    "trying to load request with unknown version");
            }
    
            ar >> command_;
            ar >> site_;
            if (command_ == command_resolve || command_ == command_unbind) {
                ar >> id_.id_;
            }
            else if (command_ == command_bind) {
                ar >> id_.id_;
                ar >> addr_.locality_;
                ar >> addr_.type_;
                ar >> addr_.address_;
            }
            else if (command_ == command_queryid || 
                     command_ == command_unregisterid) 
            {
                ar >> ns_name_;
            }
            else if (command_ == command_registerid) {
                ar >> id_.id_;
                ar >> ns_name_;
            }
        }
        BOOST_SERIALIZATION_SPLIT_MEMBER()

    private:
        boost::uint8_t command_;    /// one of the name_server_command's above
        naming::id_type id_;        /// global id (resolve, bind and unbind only)
        naming::locality site_;     /// our address 
        naming::address addr_;      /// address to associate with this id (bind only)
        std::string ns_name_;       /// namespace name (queryid only)
    };

}}}  // namespace hpx::naming::server

///////////////////////////////////////////////////////////////////////////////
// this is the current version of the parcel serialization format
// this definition needs to be in the global namespace
BOOST_CLASS_VERSION(hpx::naming::server::request, HPX_REQUEST_VERSION)

#endif
