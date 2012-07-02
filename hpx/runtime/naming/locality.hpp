//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2007 Richard D. Guidry Jr.
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_NAMING_LOCALITY_MAR_24_2008_0942AM)
#define HPX_NAMING_LOCALITY_MAR_24_2008_0942AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/safe_bool.hpp>

#include <boost/config.hpp>
#include <boost/io/ios_state.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/thread.hpp>
#include <boost/cstdint.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/iterator/filter_iterator.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/tracking.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
/// Version of locality class.
#define HPX_LOCALITY_VERSION   0x10

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming
{
    namespace detail
    {
        struct is_valid_endpoint
        {
            is_valid_endpoint(std::string const& pattern = "")
              : pattern_(pattern)
            {
            }

            bool operator()(boost::asio::ip::tcp::endpoint) const
            {
                return true;
            }

            std::string pattern_;
        };

        typedef
            boost::filter_iterator<
                is_valid_endpoint, boost::asio::ip::tcp::resolver::iterator
            >
        locality_iterator_type;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// The class \a locality is a helper class storing the address and the
    /// port number of a HPX locality.
    class HPX_EXPORT locality
    {
    public:
        locality()
          : address_(), port_(boost::uint16_t(-1))
        {}

        locality(std::string const& addr, boost::uint16_t port)
          : address_(addr), port_(port)
        {}

        locality(boost::asio::ip::address addr, boost::uint16_t port)
          : address_(addr.to_string()), port_(port)
        {}

        explicit locality(boost::asio::ip::tcp::endpoint ep)
          : address_(ep.address().to_string()), port_(ep.port())
        {}

        locality& operator= (boost::asio::ip::tcp::endpoint ep)
        {
            address_ = ep.address().to_string();
            port_ = ep.port();
            return *this;
        }

        ///////////////////////////////////////////////////////////////////////
        typedef detail::locality_iterator_type iterator_type;

        /// \brief Returns an iterator which when dereferenced will give an
        ///        endpoint suitable for a call to accept() related to this
        ///        locality
        iterator_type accept_begin(boost::asio::io_service& io_service) const;

        iterator_type accept_end() const
        {
            return locality::iterator_type();
        }

        /// \brief Returns an iterator which when dereferenced will give an
        ///        endpoint suitable for a call to connect() related to this
        ///        locality
        iterator_type connect_begin(boost::asio::io_service& io_service) const;

        iterator_type connect_end() const
        {
            return locality::iterator_type();
        }

        ///////////////////////////////////////////////////////////////////////
        friend bool operator==(locality const& lhs, locality const& rhs)
        {
            return lhs.port_ == rhs.port_ && lhs.address_ == rhs.address_;
        }

        friend bool operator!=(locality const& lhs, locality const& rhs)
        {
            return !(lhs == rhs);
        }

        friend bool operator< (locality const& lhs, locality const& rhs)
        {
            return lhs.address_ < rhs.address_ ||
                   (lhs.address_ == rhs.address_ && lhs.port_ < rhs.port_);
        }

        friend bool operator> (locality const& lhs, locality const& rhs)
        {
            return !(lhs < rhs) && !(lhs == rhs);
        }

        ///////////////////////////////////////////////////////////////////////
        operator util::safe_bool<locality>::result_type() const
        {
            return util::safe_bool<locality>()(port_ != boost::uint16_t(-1));
        }

        std::string const& get_address() const { return address_; }
        boost::uint16_t get_port() const { return port_; }

    private:
        friend std::ostream& operator<< (std::ostream& os, locality const& l);

        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void save(Archive & ar, const unsigned int version) const
        {
            ar << address_;
            ar << port_;
        }

        template<class Archive>
        void load(Archive & ar, const unsigned int version)
        {
            if (version > HPX_LOCALITY_VERSION)
            {
                HPX_THROW_EXCEPTION(version_too_new,
                    "locality::load",
                    "trying to load locality with unknown version");
            }

            ar >> address_;
            ar >> port_;
        }
        BOOST_SERIALIZATION_SPLIT_MEMBER()

    private:
        std::string address_;
        boost::uint16_t port_;
    };

    inline std::ostream& operator<< (std::ostream& os, locality const& l)
    {
        boost::io::ios_flags_saver ifs(os);
        os << l.address_ << ":" << l.port_;
        return os;
    }

///////////////////////////////////////////////////////////////////////////////
}}

///////////////////////////////////////////////////////////////////////////////
// this is the current version of the parcel serialization format
// this definition needs to be in the global namespace

#if defined(__GNUG__) && !defined(__INTEL_COMPILER)
#if defined(HPX_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#pragma GCC diagnostic push
#endif
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif
BOOST_CLASS_VERSION(hpx::naming::locality, HPX_LOCALITY_VERSION)
BOOST_CLASS_TRACKING(hpx::naming::locality, boost::serialization::track_never)
#if defined(__GNUG__) && !defined(__INTEL_COMPILER)
#if defined(HPX_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#pragma GCC diagnostic pop
#endif
#endif

#include <hpx/config/warnings_suffix.hpp>

#endif

