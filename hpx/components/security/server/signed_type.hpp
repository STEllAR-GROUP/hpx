//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_SECURITY_SERVER_SIGNED_TYPE_HPP
#define HPX_COMPONENTS_SECURITY_SERVER_SIGNED_TYPE_HPP

#include <boost/serialization/serialization.hpp>

#include "signature.hpp"

namespace hpx { namespace components { namespace security { namespace server
{
    template <typename T>
    class signed_type
    {
    public:
        signature const & get_signature() const
        {
            return signature_;
        }

        T const & get_type() const
        {
            return type_;
        }

        static signed_type invalid_signed_type;

    private:
        friend class boost::serialization::access;

        template <typename Archive>
        void serialize(Archive & ar, const unsigned int)
        {
            ar & signature_;
            ar & type_;
        }

        signature signature_;
        T type_;
    };

    template <typename T>
    signed_type<T> signed_type<T>::invalid_signed_type =
        signed_type<T>();

}}}}

#endif
