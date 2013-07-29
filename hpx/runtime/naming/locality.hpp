//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2007 Richard D. Guidry Jr.
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_NAMING_LOCALITY_MAR_24_2008_0942AM)
#define HPX_NAMING_LOCALITY_MAR_24_2008_0942AM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_PARCELPORT_MPI)
#include <hpx/util/mpi_environment.hpp>
#endif

#include <hpx/hpx_fwd.hpp>
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
#include <boost/serialization/is_bitwise_serializable.hpp>
#include <boost/serialization/array.hpp>
#include <boost/mpl/bool.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
/// Version of locality class.
#  define HPX_LOCALITY_VERSION_NO_MPI   0x10
#  define HPX_LOCALITY_VERSION_MPI      0x20

#if defined(HPX_HAVE_PARCELPORT_MPI)
#  define HPX_LOCALITY_VERSION          HPX_LOCALITY_VERSION_MPI
#else
#  define HPX_LOCALITY_VERSION          HPX_LOCALITY_VERSION_NO_MPI
#endif

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
        tcp_locality_iterator_type;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// The class \a locality is a helper class storing the address and the
    /// port number of a HPX locality.
    class HPX_EXPORT locality
    {
    public:
        locality()
          : address_(), port_(boost::uint16_t(-1))
#if defined(HPX_HAVE_PARCELPORT_MPI)
          , rank_(-1)
#endif
        {}

        locality(std::string const& addr, boost::uint16_t port)
          : address_(addr), port_(port)
#if defined(HPX_HAVE_PARCELPORT_MPI)
          , rank_(util::mpi_environment::rank())
#endif
        {}

        locality(boost::asio::ip::address addr, boost::uint16_t port)
          : address_(addr.to_string()), port_(port)
#if defined(HPX_HAVE_PARCELPORT_MPI)
          , rank_(util::mpi_environment::rank())
#endif
        {}

#if defined(HPX_HAVE_PARCELPORT_MPI)
        locality(std::string const& addr, boost::uint16_t port, int rank)
          : address_(addr), port_(port), rank_(rank)
        {}

        locality(boost::asio::ip::address addr, boost::uint16_t port, int rank)
          : address_(addr.to_string()), port_(port), rank_(rank)
        {}
#endif

        explicit locality(boost::asio::ip::tcp::endpoint ep)
          : address_(ep.address().to_string()), port_(ep.port())
#if defined(HPX_HAVE_PARCELPORT_MPI)
          , rank_(util::mpi_environment::rank())
#endif
        {}

        locality& operator= (boost::asio::ip::tcp::endpoint ep)
        {
            address_ = ep.address().to_string();
            port_ = ep.port();
#if defined(HPX_HAVE_PARCELPORT_MPI)
            rank_ = hpx::util::mpi_environment::rank();
#endif
            return *this;
        }

        ///////////////////////////////////////////////////////////////////////
        typedef detail::tcp_locality_iterator_type iterator_type;

        ///////////////////////////////////////////////////////////////////////
        friend bool operator==(locality const& lhs, locality const& rhs)
        {
#if defined(HPX_HAVE_PARCELPORT_MPI)
            if(util::mpi_environment::enabled())
                return lhs.rank_ == rhs.rank_;
#endif
            return lhs.port_ == rhs.port_ && lhs.address_ == rhs.address_;
        }

        friend bool operator!=(locality const& lhs, locality const& rhs)
        {
            return !(lhs == rhs);
        }

        friend bool operator< (locality const& lhs, locality const& rhs)
        {
#if defined(HPX_HAVE_PARCELPORT_MPI)
            if(util::mpi_environment::enabled())
                return lhs.rank_ < rhs.rank_;
#endif
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
#if defined(HPX_HAVE_PARCELPORT_MPI)
            return util::safe_bool<locality>()(
                port_ != boost::uint16_t(-1) || rank_ != -1);
#else
            return util::safe_bool<locality>()(port_ != boost::uint16_t(-1));
#endif
        }

        std::string const& get_address() const { return address_; }
        void set_address(char const* address) { address_ = address; }

        boost::uint16_t get_port() const { return port_; }
        void set_port(boost::uint16_t port) { port_ = port; }

#if defined(HPX_HAVE_PARCELPORT_MPI)
        boost::int16_t get_rank() const { return rank_; }
        void set_rank(boost::int16_t rank) { rank_ = rank; }
#else
        boost::int16_t get_rank() const { return -1; }
        void set_rank(boost::int16_t rank) {}
#endif

        parcelset::connection_type get_type() const
        {
#if defined(HPX_HAVE_PARCELPORT_MPI)
            if(rank_ != -1)
                return parcelset::connection_mpi;
#endif
            return parcelset::connection_tcpip;
        }

    private:
        friend std::ostream& operator<< (std::ostream& os, locality const& l);

        // serialization support
        friend class boost::serialization::access;

        template <typename Archive>
        void save(Archive& ar, const unsigned int version) const;

        template <typename Archive>
        void load(Archive& ar, const unsigned int version);

        BOOST_SERIALIZATION_SPLIT_MEMBER()

    private:
        std::string address_;
        boost::uint16_t port_;
#if defined(HPX_HAVE_PARCELPORT_MPI)
        boost::int16_t rank_;
#endif
    };

    inline std::ostream& operator<< (std::ostream& os, locality const& l)
    {
        boost::io::ios_flags_saver ifs(os);
        os << l.address_ << ":" << l.port_;
#if defined(HPX_HAVE_PARCELPORT_MPI)
        os << " (MPI Rank: " << l.rank_ << ")";
#endif
        return os;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns an iterator which when dereferenced will give an
    ///        endpoint suitable for a call to accept() related to this
    ///        locality
    locality::iterator_type accept_begin(locality const& loc,
        boost::asio::io_service& io_service);

    inline locality::iterator_type accept_end(locality const&)
    {
        return locality::iterator_type();
    }

    /// \brief Returns an iterator which when dereferenced will give an
    ///        endpoint suitable for a call to connect() related to this
    ///        locality
    locality::iterator_type connect_begin(locality const& loc,
        boost::asio::io_service& io_service);

    inline locality::iterator_type connect_end(locality const&) //-V524
    {
        return locality::iterator_type();
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        struct locality_serialization_data
        {
            char name_[64];         // we assume 64 bytes as an arbitrary maximum
            boost::uint16_t port_;
            boost::int16_t rank_;
        };

        inline void fill_serialization_data(locality const& l,
            detail::locality_serialization_data& data)
        {
            std::string const& address = l.get_address();
            BOOST_ASSERT(address.size() < sizeof(data.name_));
            std::size_t len = (std::min)(address.size(), sizeof(data.name_)-1);
            std::strncpy(data.name_, address.c_str(), len);
            data.name_[len] = '\0';
            data.port_ = l.get_port();
            data.rank_ = l.get_rank();
        }

        inline void fill_from_serialization_data(
            detail::locality_serialization_data const& data,
            locality& l)
        {
            l.set_address(data.name_);
            l.set_port(data.port_);
            l.set_rank(data.rank_);
        }
    }
}}

namespace boost { namespace serialization
{
    template <>
    struct is_bitwise_serializable<
            hpx::naming::detail::locality_serialization_data>
       : boost::mpl::true_
    {};
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

