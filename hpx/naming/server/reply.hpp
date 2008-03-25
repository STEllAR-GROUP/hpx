//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_NAMING_SERVER_REPLY_MAR_24_2008_0940AM)
#define HPX_NAMING_SERVER_REPLY_MAR_24_2008_0940AM

#include <vector>
#include <boost/asio.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/serialization.hpp>

#include <hpx/exception.hpp>
#include <hpx/naming/address.hpp>
#include <hpx/naming/server/request.hpp>

///////////////////////////////////////////////////////////////////////////////
///  version of GAS reply structure
#define HPX_REPLY_VERSION   0x20

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming { namespace server 
{
    /// The status of the reply.
    enum status_type
    {
        success = hpx::success,
        no_success = hpx::no_success, 
        not_implemented = hpx::not_implemented,
        out_of_memory = hpx::out_of_memory,
        internal_server_error = 4,
        service_unavailable = 5,
        bad_request = 6,
        unknown_version = 7,
        // add error codes here
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace status_strings 
    {
        char const* const success = 
            "hpx: address translation: success";
        char const* const no_success = 
            "hpx: address translation: no success";
        char const* const internal_server_error = 
            "hpx: address translation: internal server error";
        char const* const service_unavailable = 
            "hpx: address translation: service unavailable";
        char const* const bad_request = 
            "hpx: address translation: ill formatted request or unknown command";
        char const* const out_of_memory = 
            "hpx: address translation: internal server error (out of memory)";
        char const* const unknown_version = 
            "hpx: address translation: ill formatted request (unknown version)";

        inline char const* const get_error_text(status_type status)
        {
            switch (status) {
            case server::success:               return success;
            case server::no_success:            return no_success;
            case server::service_unavailable:   return service_unavailable;
            case server::bad_request:           return bad_request;
            case server::out_of_memory:         return out_of_memory;
            default:
                break;
            }
            return internal_server_error;
        }
    }
    
    ///////////////////////////////////////////////////////////////////////////
    /// a reply to be sent to a client
    class reply
    {
    public:
        reply (status_type s = server::no_success)
          : command_(command_unknown), status_(s),
            error_(status_strings::get_error_text(s)),
            prefix_(0), id_(0)
        {}

        reply (status_type s, char const* what)
          : command_(command_unknown), status_(s),
            error_(std::string(status_strings::get_error_text(s)) + ": " + what),
            prefix_(0), id_(0)
        {}

        reply (name_server_command command, status_type s = server::success)
          : command_(command), status_(s), 
            error_(status_strings::get_error_text(s)),
            prefix_(0), id_(0)
        {}

        reply (name_server_command command, naming::id_type id, 
               status_type s = server::success)
          : command_(command), status_(s),
            error_(status_strings::get_error_text(server::success)),
            prefix_(0), id_(id)
        {
            BOOST_ASSERT(s == server::success || s == server::no_success);
            BOOST_ASSERT(command == command_queryid || 
                         command == command_registerid);
        }

        reply (name_server_command command, 
               std::vector<std::pair<double, std::size_t> > const& totals, 
               status_type s = server::success)
          : command_(command), status_(s),
            error_(status_strings::get_error_text(server::success)),
            prefix_(0), id_(0)
        {
            BOOST_ASSERT(s == server::success || s == server::no_success);
            BOOST_ASSERT(command == command_statistics);
            
            for (std::size_t i = 0; i < command_lastcommand; ++i)
            {
                std::pair<double, std::size_t> const& p (totals[i]);
                if (0 != p.second)
                    statistics_[i] = p.first / p.second;
                else
                    statistics_[i] = 0.0;
            }
        }

        reply (name_server_command command, naming::address addr)
          : command_(command_resolve), status_(server::success), 
            error_(status_strings::get_error_text(server::success)),
            address_(addr),
            prefix_(0), id_(0)
        {
            BOOST_ASSERT(command == command_resolve);
        }

        reply (status_type s, name_server_command command, boost::uint64_t prefix)
          : command_(command_getprefix), status_(s),
            error_(status_strings::get_error_text(s)),
            prefix_(prefix), id_(0)
        {
            BOOST_ASSERT(s == server::success || s == server::no_success);
            BOOST_ASSERT(command == command_getprefix);
        }

        ///////////////////////////////////////////////////////////////////////
        status_type get_status() const
        {
            return (status_type)status_;
        }
        std::string get_error() const
        {
            return error_;
        }
        naming::address get_address() const
        {
            return address_;
        }
        boost::uint64_t get_prefix() const
        {
            return prefix_;
        }
        naming::id_type get_id() const
        {
            return id_;
        }
        
        double get_statictics(std::size_t i) const
        {
            if (i >= command_lastcommand)
                boost::throw_exception(hpx::error(bad_parameter));
            return statistics_[i];
        }
        
    private:
        // serialization support    
        friend class boost::serialization::access;
    
        template<class Archive>
        void save(Archive & ar, const unsigned int version) const
        {
            ar << command_;
            ar << status_;
            ar << error_;
            
            switch (command_) {
            case command_resolve:
                ar << address_.locality_;
                ar << address_.type_;
                ar << address_.address_;
                break;

            case command_getprefix:
                ar << prefix_;
                break;
                
            case command_queryid:
                ar << id_.id_;
                break;

            case command_statistics:
                for (std::size_t i = 0; i < command_lastcommand; ++i)
                    ar << statistics_[i];
                break;
                
            case command_unbind:
            case command_bind:
            case command_registerid: 
            case command_unregisterid: 
            default:
                break;  // nothing additional to be sent
            }
        }

        template<class Archive>
        void load(Archive & ar, const unsigned int version)
        {
            if (version > HPX_REPLY_VERSION) {
                throw exception(version_too_new, 
                    "trying to load reply with unknown version");
            }
    
            ar >> command_;
            ar >> status_;
            ar >> error_;
            
            switch (command_) {
            case command_resolve:
                ar >> address_.locality_;
                ar >> address_.type_;
                ar >> address_.address_;
                break;

            case command_getprefix:
                ar >> prefix_;
                break;
                
            case command_queryid:
                ar >> id_.id_;
                break;
                
            case command_statistics:
                for (std::size_t i = 0; i < command_lastcommand; ++i)
                    ar >> statistics_[i];
                break;
                
            case command_unbind:
            case command_bind:
            case command_registerid: 
            case command_unregisterid: 
            default:
                break;  // nothing additional to be received
            }
        }
        BOOST_SERIALIZATION_SPLIT_MEMBER()

    private:
        boost::uint8_t command_;    /// the command this is a reply for
        boost::uint8_t status_;     /// status of requested operation
        std::string error_;         /// descriptive error message
        naming::address address_;   /// address (for resolve only)
        boost::uint64_t prefix_;    /// prefix (for get_prefix only)
        naming::id_type id_;        /// global id (for queryid only)
        double statistics_[command_lastcommand];       /// gathered statistics
    };

///////////////////////////////////////////////////////////////////////////////
}}}  // namespace hpx::naming::server

///////////////////////////////////////////////////////////////////////////////
// this is the current version of the parcel serialization format
// this definition needs to be in the global namespace
BOOST_CLASS_VERSION(hpx::naming::server::reply, HPX_REPLY_VERSION)

#endif 
