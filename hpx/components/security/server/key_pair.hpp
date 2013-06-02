//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_SECURITY_SERVER_KEY_PAIR_HPP
#define HPX_COMPONENTS_SECURITY_SERVER_KEY_PAIR_HPP

#include <boost/serialization/serialization.hpp>

#include "public_key.hpp"
#include "secret_key.hpp"
#include "signed_type.hpp"

namespace hpx { namespace components { namespace security { namespace server
{
    class key_pair
    {
    public:
        key_pair()
          : secret_key_(public_key_)
        {
        }

        public_key const & get_public_key() const
        {
            return public_key_;
        }

        template <typename T>
        signed_type<T> sign(T const & type) const
        {
            return secret_key_.sign(type);
        }

        template <typename T>
        bool verify(signed_type<T> const & signed_type) const
        {
            return public_key_.verify(signed_type);
        }

    private:
        friend class boost::serialization::access;

        template <typename Archive>
        void serialize(Archive & ar, const unsigned int)
        {
            ar & public_key_;
            ar & secret_key_;
        }

        public_key public_key_;
        secret_key secret_key_;
    };
}}}}

#endif
