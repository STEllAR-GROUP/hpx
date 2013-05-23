//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_SECURITY_SERVER_CERTIFICATE_SIGNING_REQUEST_HPP
#define HPX_COMPONENTS_SECURITY_SERVER_CERTIFICATE_SIGNING_REQUEST_HPP

#include <boost/serialization/serialization.hpp>
#include <hpx/hpx_fwd.hpp>

#include "capability.hpp"
#include "identity.hpp"
#include "public_key.hpp"

namespace hpx { namespace components { namespace security { namespace server
{
    class certificate_signing_request
    {
    public:
        static std::size_t const capability_size = 8;

        certificate_signing_request()
        {
        }

        certificate_signing_request(identity const & subject,
                                    public_key const & subject_public_key)
          : subject_(subject)
          , subject_public_key_(subject_public_key)
        {
        }

        identity const & get_subject() const
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
            ar & subject_;
            ar & subject_public_key_;
            ar & capability_;
        }

        identity subject_;
        public_key subject_public_key_;

        capability capability_;
    };
}}}}

#endif
