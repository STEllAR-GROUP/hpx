#ifndef HPX_COMPONENTS_SECURITY_SERVER_IDENTITY_HPP
#define HPX_COMPONENTS_SECURITY_SERVER_IDENTITY_HPP

#include <boost/cstdint.hpp>
#include <boost/serialization/serialization.hpp>
#include <hpx/hpx_fwd.hpp>

namespace hpx { namespace components { namespace security { namespace server
{
    class identity
    {
    public:
        identity()
          : msb_(0), lsb_(0)
        {
        }

        identity(boost::uint64_t msb, boost::uint64_t lsb)
          : msb_(msb), lsb_(lsb)
        {
        }

        identity(hpx::naming::id_type const & id)
          : msb_(id.get_msb()), lsb_(id.get_lsb())
        {
        }

        boost::uint64_t get_msb() const
        {
            return msb_;
        }

        boost::uint64_t get_lsb() const
        {
            return lsb_;
        }

    private:
        friend class boost::serialization::access;

        template <typename Archive>
        void serialize(Archive & ar, const unsigned int)
        {
            ar & msb_;
            ar & lsb_;
        }

        boost::uint64_t msb_;
        boost::uint64_t lsb_;
    };
}}}}

#endif
