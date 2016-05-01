//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_SECURITY_SERVER_CERTIFICATE_STORE_HPP
#define HPX_COMPONENTS_SECURITY_SERVER_CERTIFICATE_STORE_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_SECURITY)
#include <hpx/error_code.hpp>
#include <hpx/throw_exception.hpp>

#include <boost/optional.hpp>
#include <boost/format.hpp>

#include "certificate.hpp"
#include "public_key.hpp"
#include "signed_type.hpp"

#include <map>
#include <sstream>

namespace hpx { namespace components { namespace security
{
    class certificate_store
    {
        HPX_NON_COPYABLE(certificate_store);

        typedef std::map<
            naming::gid_type, signed_type<certificate>
        > store_type;

    public:
        certificate_store(signed_type<certificate> const & signed_certificate)
        {
            certificate const & certificate = signed_certificate.get_type();

            if (certificate.get_issuer() != certificate.get_subject())
            {
                HPX_THROW_EXCEPTION(
                    hpx::security_error
                  , "certificate_store::certificate_store"
                  , boost::str(boost::format(
                        "The certificate is not self-signed: %1%") %
                        signed_certificate)
                )
            }

            public_key const & issuer_public_key =
                certificate.get_subject_public_key();
            if (!issuer_public_key.verify(signed_certificate))
            {
                HPX_THROW_EXCEPTION(
                    hpx::security_error
                  , "certificate_store::certificate_store"
                  , boost::str(boost::format(
                        "The certificate signature is invalid: %1%") %
                        signed_certificate)
                )
            }

            store_.insert(
                std::make_pair(certificate.get_subject(), signed_certificate));
        }

        void insert(signed_type<certificate> const & signed_certificate)
        {
            certificate const & certificate = signed_certificate.get_type();

            naming::gid_type const & subject = certificate.get_subject();

            // FIXME, expiration dates?
            if (store_.find(subject) != store_.end())
                return;

            store_type::const_iterator issuer =
                store_.find(certificate.get_issuer());
            if (issuer == store_.end())
            {
                std::ostringstream oss;
                oss << boost::str(boost::format(
                    "The certificate issuer is unknown: %1%\n") % subject);
                oss << "Known certificate subjects:\n";

                store_type::const_iterator end = store_.end();
                for(store_type::const_iterator it = store_.begin(); it != end; ++it)
                {
                    oss << "    " << it->first << "\n";
                }

                HPX_THROW_EXCEPTION(
                    hpx::security_error
                  , "certificate_store::insert"
                  , oss.str()
                )
            }

            public_key const & issuer_public_key =
                issuer->second.get_type().get_subject_public_key();
            if (issuer_public_key.verify(signed_certificate) == false)
            {
                HPX_THROW_EXCEPTION(
                    hpx::security_error
                  , "certificate_store::insert"
                  , boost::str(boost::format(
                        "The certificate signature is invalid: %1%") %
                        signed_certificate)
                )
            }

            capability const & issuer_capability =
                issuer->second.get_type().get_capability();
            if (issuer_capability.verify_delegation(
                    certificate.get_capability()) == false)
            {
                HPX_THROW_EXCEPTION(
                    hpx::security_error
                  , "certificate_store::insert"
                  , boost::str(boost::format(
                      "The issuer can't delegated the requested capabilities: %1% %2%") %
                      issuer_capability % certificate.get_capability())
                )
            }

            store_.insert(std::make_pair(subject, signed_certificate));
        }

        signed_type<certificate> const&
        at(naming::gid_type const & subject, error_code& ec = throws) const
        {
            store_type::const_iterator iterator = store_.find(subject);

            if (iterator == store_.end())
            {
                std::ostringstream strm;
                strm << boost::str(boost::format(
                    "requesting a certificate for an unknown subject: %1%\n") %
                    subject);
                strm << "  known certificate subjects:\n";

                store_type::const_iterator end = store_.end();
                for(store_type::const_iterator it = store_.begin(); it != end; ++it)
                {
                    strm << "    " << (*it).first << "\n";
                }

                HPX_THROWS_IF(
                    ec, hpx::security_error
                  , "certificate_store::certificate_store"
                  , strm.str()
                );

                return signed_type<certificate>::invalid_signed_type;
            }

            if (&ec != &throws)
                ec = make_success_code();

            return iterator->second;
        }

        signed_type<certificate> const&
        at_locality(naming::gid_type const & gid, error_code& ec = throws) const
        {
            naming::gid_type subject(
                hpx::naming::replace_locality_id(
                    HPX_SUBORDINATE_CERTIFICATE_AUTHORITY_MSB
                  , get_locality_id_from_gid(gid))
              , HPX_SUBORDINATE_CERTIFICATE_AUTHORITY_LSB);

            return at(subject, ec);
        }

        signed_type<certificate> const&
        at_locality(boost::uint32_t locality_id, error_code& ec = throws) const
        {
            naming::gid_type subject(
                hpx::naming::replace_locality_id(
                    HPX_SUBORDINATE_CERTIFICATE_AUTHORITY_MSB
                  , locality_id)
              , HPX_SUBORDINATE_CERTIFICATE_AUTHORITY_LSB);

            return at(subject, ec);
        }

    private:
        store_type store_;
    };
}}}

#endif

#endif
