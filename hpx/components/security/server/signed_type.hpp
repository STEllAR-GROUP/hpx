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
#if defined(_MSC_VER)
#  pragma pack(push, 1)
#endif

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

        operator T const & () const
        {
            return type_;
        }

        static signed_type invalid_signed_type;

        friend std::ostream & operator<<(std::ostream & os,
                                         signed_type<T> const & signed_type)
        {
            return os << "<signed_type "
                      << signed_type.signature_
                      << " "
                      << signed_type.type_
                      << ">";
        }

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

#if defined(_MSC_VER)
#  pragma pack(pop)
#endif

    template <typename T>
    signed_type<T> signed_type<T>::invalid_signed_type =
        signed_type<T>();

}}}}

#endif
