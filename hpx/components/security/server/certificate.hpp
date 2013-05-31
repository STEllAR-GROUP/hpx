#ifndef HPX_SECURITY_CERTIFICATE_HPP
#define HPX_SECURITY_CERTIFICATE_HPP

#include <boost/serialization/serialization.hpp>
#include <hpx/runtime/naming/name.hpp>

#include "certificate_signing_request.hpp"
#include "public_key.hpp"

namespace hpx { namespace components { namespace security { namespace server
{
    class certificate
    {
    public:
        certificate()
        {
        }

        certificate(naming::gid_type const & issuer,
                    naming::gid_type const & subject,
                    public_key const & subject_public_key,
                    capability const & capability)
          : issuer_(issuer)
          , subject_(subject)
          , subject_public_key_(subject_public_key)
          , capability_(capability)
        {
        }

        certificate(naming::gid_type const & issuer,
                    certificate_signing_request const & csr)
          : issuer_(issuer)
          , subject_(csr.get_subject())
          , subject_public_key_(csr.get_subject_public_key())
          , capability_(csr.get_capability())
        {
        }

        naming::gid_type const & get_issuer() const
        {
            return issuer_;
        }

        naming::gid_type const & get_subject() const
        {
            return subject_;
        }

        public_key const & get_subject_public_key() const
        {
            return subject_public_key_;
        }

        capability const & get_capability() const
        {
            return capability_;
        }

    private:
        friend class boost::serialization::access;

        template <typename Archive>
        void serialize(Archive & ar, const unsigned int)
        {
            ar & issuer_;

            ar & subject_;
            ar & subject_public_key_;

            ar & capability_;
        }

        naming::gid_type issuer_;

        naming::gid_type subject_;
        public_key subject_public_key_;

        capability capability_;
    };
}}}}

#endif
