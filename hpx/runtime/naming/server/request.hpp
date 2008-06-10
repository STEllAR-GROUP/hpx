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

#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/naming/locality.hpp>

///////////////////////////////////////////////////////////////////////////////
///  version of GAS request structure
#define HPX_REQUEST_VERSION   0x30

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
        command_bind_range = 2,     /// bind a range of addresses to a range of global ids
        command_unbind_range = 3,   /// remove binding for a range of global ids
        command_resolve = 4,        /// resolve a global id to an address
        command_queryid = 5,        /// query for a global id associated with a namespace name (string)
        command_registerid = 6,     /// associate a namespace name with a global id
        command_unregisterid = 7,   /// remove association of a namespace name with a global id
        command_statistics_count = 8,    /// return some usage statistics: execution count 
        command_statistics_mean = 9,     /// return some usage statistics: average server execution time
        command_statistics_moment2 = 10, /// return some usage statistics: 2nd moment of server execution time
        command_lastcommand
    };

    const char* const command_names[] = 
    {
        "command_getprefix",
        "command_getidrange",
        "command_bind_range",
        "command_unbind_range",
        "command_resolve",
        "command_queryid",
        "command_registerid",
        "command_unregisterid",
        "command_statistics_count",
        "command_statistics_mean",
        "command_statistics_moment2",
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
          : command_(c)
        {}
        
        // get_prefix
        request(name_server_command c, locality const& l) 
          : command_(c), site_(l)
        {}
        
        // resolve
        request(name_server_command c, naming::id_type const& id) 
          : command_(c), id_(id)
        {}

        // registerid
        request(name_server_command c, std::string const& ns_name, 
                naming::id_type const& id) 
          : command_(c), id_(id), ns_name_(ns_name)
        {}

        // unregisterid
        request(name_server_command c, std::string const& ns_name) 
          : command_(c), ns_name_(ns_name)
        {}

        // bind_range
        request(name_server_command c, naming::id_type id, std::size_t count, 
                address const& addr, std::ptrdiff_t offset) 
          : command_(c), id_(id), count_(count), addr_(addr), offset_(offset)
        {}

        // unbind_range
        request(name_server_command c, naming::id_type id, std::size_t count) 
          : command_(c), id_(id), count_(count)
        {}

        boost::uint8_t get_command() const
        {
            return command_;
        }
        
        naming::id_type const& get_id() const
        {
            return id_;
        }
        
        std::size_t const& get_count() const
        {
            return count_;
        }
        
        naming::locality get_site() const
        {
            return site_;
        }
        
        naming::address get_address() const 
        {
            return addr_;
        }
        
        std::ptrdiff_t const& get_offset() const
        {
            return offset_;
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
                ar << id_;
                break;

            case command_bind_range:
                ar << id_;
                ar << count_;
                ar << addr_.locality_;
                ar << addr_.type_;
                ar << addr_.address_;
                ar << offset_;
                break;

            case command_unbind_range:
                ar << id_;
                ar << count_;
                break;

            case command_queryid: 
            case command_unregisterid:
                ar << ns_name_;
                break;

            case command_registerid:
                ar << id_;
                ar << ns_name_;
                break;

            case command_getprefix:
            case command_getidrange:
            case command_statistics_count:
            case command_statistics_mean:
            case command_statistics_moment2:
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
                ar >> id_;
                break;

            case command_bind_range:
                ar >> id_;
                ar >> count_;
                ar >> addr_.locality_;
                ar >> addr_.type_;
                ar >> addr_.address_;
                ar >> offset_;
                break;

            case command_unbind_range:
                ar >> id_;
                ar >> count_;
                break;

            case command_queryid:
            case command_unregisterid: 
                ar >> ns_name_;
                break;

            case command_registerid:
                ar >> id_;
                ar >> ns_name_;
                break;

            case command_getprefix:
            case command_getidrange:
            case command_statistics_count:
            case command_statistics_mean:
            case command_statistics_moment2:
            default:
                // nothing additional to be received
                break;
            }
        }
        BOOST_SERIALIZATION_SPLIT_MEMBER()

    private:
        boost::uint8_t command_;    /// one of the name_server_command's above
        naming::id_type id_;        /// global id (resolve, bind and unbind only)
        std::size_t count_;         /// number of global ids (bind_range, unbind_range only)
        naming::locality site_;     /// our address 
        naming::address addr_;      /// address to associate with this id (bind only)
        std::ptrdiff_t offset_;     /// offset between local addresses of a range (bind_range only)
        std::string ns_name_;       /// namespace name (queryid only)
    };

    // debug support for a request class
    inline std::ostream& operator<< (std::ostream& os, request const& req)
    {
        os << get_command_name(req.command_) << ": ";

        switch (req.command_) {
        case command_resolve:
            os << "id:" << std::hex << req.id_ << " ";
            break;

        case command_bind_range:
            os << "id:" << std::hex << req.id_ << " ";
            if (req.count_ != 1)
                os << "count:" << std::dec << req.count_ << " ";
            os << "addr(" << req.addr_ << ") ";
            if (req.offset_ != 0)
                os << "offset:" << std::dec << req.offset_ << " ";
            break;

        case command_unbind_range:
            os << "id:" << std::hex << req.id_ << " ";
            if (req.count_ != 1)
                os << "count:" << std::dec << req.count_ << " ";
            break;

        case command_queryid:
        case command_unregisterid: 
            os << "name(\"" << req.ns_name_ << "\") ";
            break;

        case command_registerid:
            os << "id:" << std::hex << req.id_ << " ";
            os << "name(\"" << req.ns_name_ << "\") ";
            break;

        case command_getprefix:
        case command_getidrange:
            os << "site(" << req.site_ << ") ";
            break;
            
        case command_statistics_count:
        case command_statistics_mean:
        case command_statistics_moment2:
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
