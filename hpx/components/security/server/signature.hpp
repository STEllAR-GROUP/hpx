//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_SECURITY_SERVER_SIGNATURE_HPP
#define HPX_COMPONENTS_SECURITY_SERVER_SIGNATURE_HPP

#include <boost/array.hpp>
#include <boost/serialization/serialization.hpp>
#include <sodium.h>

namespace hpx { namespace components { namespace security { namespace server
{
    class signature
    {
    private:
        friend class boost::serialization::access;

        template <typename Archive>
        void serialize(Archive & ar, const unsigned int)
        {
            ar & bytes_;
        }

        boost::array<
            unsigned char, crypto_sign_BYTES
        > bytes_;
    };
}}}}

#endif
