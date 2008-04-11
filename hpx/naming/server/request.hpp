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
        command_unknown = -1,
        command_firstcommand = 0,
        command_getprefix = 0,      /// return a unique prefix for the requesting site
        command_getidrange = 1,     /// return a unique range of ids for the requesting site
        command_bind = 2,           /// bind an address to a global id
        command_unbind = 3,         /// remove binding for a global id
        command_resolve = 4,        /// resolve a global id to an address
        command_queryid = 5,        /// query for a global id associated with a namespace name (string)
        command_registerid = 6,     /// associate a namespace name with a global id
        command_unregisterid = 7,   /// remove association of a namespace name with a global id
        command_statistics = 8,     /// return some usage statistics
        command_lastcommand
    };

    const char* const command_names[] = 
    {
        "command_getprefix",
        "command_getidrange",
        "command_bind",
        "command_unbind",
        "command_resolve",
        "command_queryid",
        "command_registerid",
        "command_unregisterid",
        "command_statistics",
        ""
    };
    
    inline char const* const get_command_name(int cmd)
    {
        if (cmd >= command_firstcommand && cmd < command_lastcommand)
            return command_names[cmd];
        return "<unknown>";
    }
    
    /// A request received from a client.
    class request
    {
    public:
        request(name_server_command c = command_unknown) 
          : command_(c), id_(0)
        {}
        
        request(name_server_command c, locality const& l) 
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
        friend std::ostream& operator<< (std::ostream& os, request const& req);
        
        // serialization support    
        friend class boost::serialization::access;
    
        template<class Archive>
        void save(Archive & ar, const unsigned int version) const
        {
            ar << command_;
            ar << site_;
            
            switch (command_) {
            case command_resolve:
            case command_unbind:
                ar << id_.id_;
                break;

            case command_bind:
                ar << id_.id_;
                ar << addr_.locality_;
                ar << addr_.type_;
                ar << addr_.address_;
                break;

            case command_queryid: 
            case command_unregisterid:
                ar << ns_name_;
                break;

            case command_registerid:
                ar << id_.id_;
                ar << ns_name_;
                break;

            case command_getprefix:
            case command_getidrange:
            case command_statistics:
            default:
                // nothing additional to be sent
                break;
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
            
            switch (command_) {
            case command_resolve:
            case command_unbind:
                ar >> id_.id_;
                break;

            case command_bind:
                ar >> id_.id_;
                ar >> addr_.locality_;
                ar >> addr_.type_;
                ar >> addr_.address_;
                break;

            case command_queryid:
            case command_unregisterid: 
                ar >> ns_name_;
                break;

            case command_registerid:
                ar >> id_.id_;
                ar >> ns_name_;
                break;

            case command_getprefix:
            case command_getidrange:
            case command_statistics:
            default:
                // nothing additional to be received
                break;
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

    // debug support for a request class
    inline std::ostream& operator<< (std::ostream& os, request const& req)
    {
        os << get_command_name(req.command_) << ": ";

        switch (req.command_) {
        case command_unbind:
        case command_resolve:
            os << "id(" << std::hex << req.id_.id_ << ") ";
            break;

        case command_bind:
            os << "id(" << std::hex << req.id_.id_ << ") ";
            os << "addr(" << req.addr_ << ") ";
            break;

        case command_queryid:
        case command_unregisterid: 
            os << "name(" << req.ns_name_ << ") ";
            break;

        case command_registerid:
            os << "id(" << std::hex << req.id_.id_ << ") ";
            os << "name(" << req.ns_name_ << ") ";
            break;

        case command_getprefix:
        case command_getidrange:
            os << "site(" << req.site_ << ") ";
            break;
            
        case command_statistics:
        default:
            break;
        }
        return os;
    }

}}}  // namespace hpx::naming::server

///////////////////////////////////////////////////////////////////////////////
// this is the current version of the parcel serialization format
// this definition needs to be in the global namespace
BOOST_CLASS_VERSION(hpx::naming::server::request, HPX_REQUEST_VERSION)

#endif
