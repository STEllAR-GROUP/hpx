//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_NAMING_SERVER_REPLY_MAR_24_2008_0940AM)
#define HPX_NAMING_SERVER_REPLY_MAR_24_2008_0940AM

#include <vector>
#include <string>
#include <iosfwd>

#include <boost/serialization/split_member.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/version.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/server/request.hpp>

///////////////////////////////////////////////////////////////////////////////
///  version of GAS reply structure
#define HPX_REPLY_VERSION   0x30

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming { namespace server 
{
    ///////////////////////////////////////////////////////////////////////////
    /// The function \a get_error_text returns the textual representation 
    /// of the passed error code.
    char const* const get_error_text(error status);

    ///////////////////////////////////////////////////////////////////////////
    /// a reply to be sent to a client
    class reply
    {
    public:
        reply (error s = no_success)
          : command_(command_unknown), status_(s),
            error_(get_error_text(s)),
            lower_bound_(0), upper_bound_(0)
        {}

        reply (error s, char const* what)
          : command_(command_unknown), status_(s),
            error_(get_error_text(s)),
            lower_bound_(0), upper_bound_(0)
        {
            error_ += std::string(": ") + what;
        }

        reply (agas_server_command command, error s = success,
                char const* what = 0)
          : command_(command), status_(s), 
            error_(get_error_text(s)),
            lower_bound_(0), upper_bound_(0)
        {
            if (0 != what)
                error_ += std::string(": ") + what;
        }

        reply (agas_server_command command, naming::id_type const& id, 
               error s = success)
          : command_(command), status_(s),
            error_(get_error_text(s)),
            lower_bound_(0), upper_bound_(0), id_(id)
        {
            BOOST_ASSERT(s == success || s == no_success);
            BOOST_ASSERT(command == command_queryid || 
                         command == command_registerid);
        }

        reply (agas_server_command command, components::component_type type, 
                error s = success)
          : command_(command), status_(s), error_(get_error_text(s)), 
            type_(type)
        {
            BOOST_ASSERT(command == command_get_component_id || 
                         command == command_register_factory);
        }

        reply (std::vector<boost::uint32_t>& prefixes, error s = success)
          : command_(command_getprefixes), status_(s),
            error_(get_error_text(s))
        {
            BOOST_ASSERT(s == success || s == no_success);
            std::swap(prefixes_, prefixes);
        }

        template <typename Container, typename F>
        reply (agas_server_command command, Container const& totals, F f,
               error s = success)
          : command_(command), status_(s),
            error_(get_error_text(s)),
            lower_bound_(0), upper_bound_(0)
        {
            BOOST_ASSERT(s == success || s == no_success);
            BOOST_ASSERT(command == command_statistics_count ||
                         command == command_statistics_mean ||
                         command == command_statistics_moment2);
            
            for (std::size_t i = 0; i < command_lastcommand; ++i)
                statistics_.push_back(double(f(totals[i])));
        }

        reply (agas_server_command command, naming::address addr)
          : command_(command), status_(success), 
            error_(get_error_text(success)),
            address_(addr),
            lower_bound_(0), upper_bound_(0)
        {
            BOOST_ASSERT(command == command_resolve ||
                         command == command_unbind_range);
        }

        reply (error s, agas_server_command command, 
                naming::id_type prefix)
          : command_(command), status_(s),
            error_(get_error_text(s)),
            lower_bound_(prefix), upper_bound_(0)
        {
            BOOST_ASSERT(s == success || s == repeated_request);
            BOOST_ASSERT(command == command_getprefix || 
                command == command_getconsoleprefix);
        }

        reply (error s, agas_server_command command, 
                naming::id_type lower_bound, naming::id_type upper_bound)
          : command_(command_getidrange), status_(s),
            error_(get_error_text(s)),
            lower_bound_(lower_bound), upper_bound_(upper_bound)
        {
            BOOST_ASSERT(s == success || s == repeated_request);
            BOOST_ASSERT(command == command_getidrange);
        }

        ///////////////////////////////////////////////////////////////////////
        error get_status() const
        {
            return (error)status_;
        }
        std::string get_error() const
        {
            return error_;
        }
        naming::address get_address() const
        {
            return address_;
        }
        naming::id_type const& get_prefix() const
        {
            return lower_bound_;
        }
        naming::id_type const& get_lower_bound() const
        {
            return lower_bound_;
        }
        naming::id_type const& get_upper_bound() const
        {
            return upper_bound_;
        }
        naming::id_type get_id() const
        {
            return id_;
        }

        components::component_type get_component_id() const
        {
            return type_;
        }

        double get_statictics(std::size_t i) const
        {
            if (i >= command_lastcommand)
                HPX_THROW_EXCEPTION(bad_parameter, 
                    "reply::get_statictics", 
                    "unknown AGAS command");
            return statistics_[i];
        }

        std::vector<double>& get_statictics() 
        {
            return statistics_;
        }

        std::vector<boost::uint32_t> const& get_prefixes() const
        {
            return prefixes_;
        }

    private:
        friend std::ostream& operator<< (std::ostream& os, reply const& rep);

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
            case command_unbind_range:
                ar << address_.locality_;
                ar << address_.type_;
                ar << address_.address_;
                break;

            case command_getidrange:
                ar << upper_bound_;
                // fall through

            case command_getconsoleprefix:
            case command_getprefix:
                ar << lower_bound_;
                break;

            case command_getprefixes:
                ar << prefixes_;
                break;

            case command_queryid:
                ar << id_;
                break;

            case command_get_component_id:
            case command_register_factory:
                ar << type_;
                break;

            case command_statistics_count:
            case command_statistics_mean:
            case command_statistics_moment2:
                ar << statistics_;
                break;

            case command_bind_range:
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
            case command_unbind_range:
                ar >> address_.locality_;
                ar >> address_.type_;
                ar >> address_.address_;
                break;

            case command_getidrange:
                ar >> upper_bound_;
                // fall through
            case command_getconsoleprefix:
            case command_getprefix:
                ar >> lower_bound_;
                break;

            case command_getprefixes:
                ar >> prefixes_;
                break;

            case command_queryid:
                ar >> id_;
                break;

            case command_get_component_id:
            case command_register_factory:
                ar >> type_;
                break;

            case command_statistics_count:
            case command_statistics_mean:
            case command_statistics_moment2:
                ar >> statistics_;
                break;

            case command_bind_range:
            case command_registerid: 
            case command_unregisterid: 
            default:
                break;  // nothing additional to be received
            }
        }
        BOOST_SERIALIZATION_SPLIT_MEMBER()

    private:
        boost::uint8_t command_;        ///< the command this is a reply for
        boost::uint8_t status_;         ///< status of requested operation
        std::string error_;             ///< descriptive error message
        naming::address address_;       ///< address (for resolve only)
        naming::id_type lower_bound_;   ///< lower bound of id range (for get_idrange only) 
                                        ///< or the prefix for the given locality (get_prefix only)
        naming::id_type upper_bound_;   ///< upper bound of id range (for get_idrange only)
        naming::id_type id_;            ///< global id (for queryid only)
        std::vector<double> statistics_;        ///< gathered statistics
        std::vector<boost::uint32_t> prefixes_; ///< all site prefixes known to this server
        components::component_type type_;       ///< the component type (command_get_component_id, command_register_factory only)
    };

    /// The \a operator<< is used for logging purposes, dumping the internal 
    /// data structures of a \a reply instance to the given ostream.
    std::ostream& operator<< (std::ostream& os, reply const& rep);

///////////////////////////////////////////////////////////////////////////////
}}}  // namespace hpx::naming::server

///////////////////////////////////////////////////////////////////////////////
// this is the current version of the parcel serialization format
// this definition needs to be in the global namespace
BOOST_CLASS_VERSION(hpx::naming::server::reply, HPX_REPLY_VERSION)
BOOST_CLASS_TRACKING(hpx::naming::server::reply, boost::serialization::track_never)

#endif 
