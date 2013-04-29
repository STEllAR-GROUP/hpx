//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_SECURITY_SERVER_CERTIFICATE_SIGNING_REQUEST_HPP
#define HPX_COMPONENTS_SECURITY_SERVER_CERTIFICATE_SIGNING_REQUEST_HPP

#include <hpx/hpx_fwd.hpp>
#include <boost/serialization/serialization.hpp>

#include "public_key.hpp"

namespace hpx { namespace components { namespace security { namespace server
{
    class certificate_signing_request
    {
    public:
        certificate_signing_request()
        {
        }

        certificate_signing_request(naming::id_type const & subject,
                                    public_key const & subject_public_key)
          : subject_(subject)
          , subject_public_key_(subject_public_key)
        {
        }

        naming::id_type const & get_subject() const
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
            ar & subject_;
            ar & subject_public_key_;
        }

        naming::id_type subject_;
        public_key subject_public_key_;
    };
}}}}

#endif
