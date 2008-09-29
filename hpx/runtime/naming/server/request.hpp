//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_NAMING_SERVER_REQUEST_MAR_24_2008_0941AM)
#define HPX_NAMING_SERVER_REQUEST_MAR_24_2008_0941AM

#include <iosfwd>
#include <string>

#include <boost/cstdint.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/serialization.hpp>

#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/locality.hpp>

///////////////////////////////////////////////////////////////////////////////
///  version of GAS request structure
#define HPX_REQUEST_VERSION   0x30

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming { namespace server 
{
    /// the commands supported by the DGAS server
    enum dgas_server_command
    {
        command_unknown = -1,
        command_firstcommand = 0,
        command_getprefix = 0,      ///< return a unique prefix for the requesting site
        command_getprefixes = 1,    ///< return prefixes for all known localities in the system
        command_get_component_id = 2,    ///< return an unique component type
        command_getidrange = 3,     ///< return a unique range of ids for the requesting site
        command_bind_range = 4,     ///< bind a range of addresses to a range of global ids
        command_unbind_range = 5,   ///< remove binding for a range of global ids
        command_resolve = 6,        ///< resolve a global id to an address
        command_queryid = 7,        ///< query for a global id associated with a namespace name (string)
        command_registerid = 8,     ///< associate a namespace name with a global id
        command_unregisterid = 9,   ///< remove association of a namespace name with a global id
        command_statistics_count = 10,   ///< return some usage statistics: execution count 
        command_statistics_mean = 11,    ///< return some usage statistics: average server execution time
        command_statistics_moment2 = 12, ///< return some usage statistics: 2nd moment of server execution time
        command_lastcommand
    };

    ///////////////////////////////////////////////////////////////////////////
    char const* const get_command_name(int cmd);

    /// A request received from a client.
    class request
    {
    public:
        request(dgas_server_command c = command_unknown) 
          : command_(c)
        {}

        // get_prefix
        request(dgas_server_command c, locality const& l) 
          : command_(c), site_(l)
        {}

        // get_id_range
        request(dgas_server_command c, locality const& l, std::size_t count) 
          : command_(c), count_(count), site_(l)
        {}

        // resolve
        request(dgas_server_command c, naming::id_type const& id) 
          : command_(c), id_(id)
        {}

        // registerid
        request(dgas_server_command c, std::string const& ns_name, 
                naming::id_type const& id) 
          : command_(c), id_(id), name_(ns_name)
        {}

        // get_component_id
        // unregisterid
        request(dgas_server_command c, std::string const& ns_name) 
          : command_(c), name_(ns_name)
        {}

        // bind_range
        request(dgas_server_command c, naming::id_type id, std::size_t count, 
                address const& addr, std::ptrdiff_t offset) 
          : command_(c), id_(id), count_(count), 
            addr_(addr), offset_(offset)
        {}

        // unbind_range
        request(dgas_server_command c, naming::id_type id, std::size_t count) 
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
            return name_;
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

            case command_get_component_id:
            case command_queryid: 
            case command_unregisterid:
                ar << name_;
                break;

            case command_registerid:
                ar << id_;
                ar << name_;
                break;

            case command_getidrange:
                ar << count_;
                break;

            case command_getprefix:
            case command_getprefixes:
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

            case command_get_component_id:
            case command_queryid:
            case command_unregisterid: 
                ar >> name_;
                break;

            case command_registerid:
                ar >> id_;
                ar >> name_;
                break;

            case command_getidrange:
                ar >> count_;
                break;

            case command_getprefix:
            case command_getprefixes:
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
        boost::uint8_t command_;    /// one of the dgas_server_command's above
        naming::id_type id_;        /// global id (resolve, bind and unbind only)
        std::size_t count_;         /// number of global ids (bind_range, unbind_range only)
        naming::locality site_;     /// our address 
        naming::address addr_;      /// address to associate with this id (bind only)
        std::ptrdiff_t offset_;     /// offset between local addresses of a range (bind_range only)
        std::string name_;       /// namespace name (queryid only)
    };

    /// The \a operator<< is used for logging purposes, dumping the internal 
    /// data structures of a \a request instance to the given ostream.
    std::ostream& operator<< (std::ostream& os, request const& req);

}}}  // namespace hpx::naming::server

///////////////////////////////////////////////////////////////////////////////
// this is the current version of the parcel serialization format
// this definition needs to be in the global namespace
BOOST_CLASS_VERSION(hpx::naming::server::request, HPX_REQUEST_VERSION)
BOOST_CLASS_TRACKING(hpx::naming::server::request, boost::serialization::track_never)

#endif
