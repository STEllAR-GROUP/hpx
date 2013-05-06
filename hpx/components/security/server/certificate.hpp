#ifndef HPX_SECURITY_CERTIFICATE_HPP
#define HPX_SECURITY_CERTIFICATE_HPP

#include <boost/serialization/serialization.hpp>

#include "certificate_signing_request.hpp"
#include "identity.hpp"
#include "public_key.hpp"

namespace hpx { namespace components { namespace security { namespace server
{
    class certificate
    {
    public:
        certificate()
        {
        }

        certificate(identity const & issuer,
                    identity const & subject,
                    public_key const & subject_public_key)
          : issuer_(issuer)
          , subject_(subject)
          , subject_public_key_(subject_public_key)
        {
        }

        certificate(identity const & issuer,
                    certificate_signing_request const & csr)
          : issuer_(issuer)
          , subject_(csr.get_subject())
          , subject_public_key_(csr.get_subject_public_key())
        {
        }

        identity const & get_issuer() const
        {
            return issuer_;
        }

        identity const & get_subject() const
        {
            return subject_;
        }

        public_key const & get_subject_public_key() const
        {
            return subject_public_key_;
        }

    private:
        friend class boost::serialization::access;

        template <typename Archive>
        void serialize(Archive & ar, const unsigned int)
        {
            ar & issuer_;

            ar & subject_;
            ar & subject_public_key_;
        }

        identity issuer_;

        identity subject_;
        public_key subject_public_key_;
    };
}}}}

#endif
