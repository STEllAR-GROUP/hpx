//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_SECURITY_SERVER_SIGNED_TYPE_HPP
#define HPX_COMPONENTS_SECURITY_SERVER_SIGNED_TYPE_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_SECURITY)

#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/traits/is_bitwise_serializable.hpp>

#include <hpx/components/security/signature.hpp>

#include <cstddef>
#include <type_traits>

namespace hpx { namespace components { namespace security
{
#if defined(HPX_MSVC)
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

        unsigned char* begin()
        {
            return reinterpret_cast<unsigned char*>(this);
        }
        unsigned char const* begin() const
        {
            return reinterpret_cast<unsigned char const*>(this);
        }

        unsigned char* end()
        {
            return reinterpret_cast<unsigned char*>(this) + size();
        }
        unsigned char const* end() const
        {
            return reinterpret_cast<unsigned char const*>(this) + size();
        }

        HPX_CONSTEXPR static std::size_t size()
        {
            return sizeof(signed_type);
        }

    private:
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive & ar, const unsigned int)
        {
            ar & hpx::serialization::make_array(begin(), size());
        }

        signature signature_;
        T type_;
    };

#if defined(HPX_MSVC)
#  pragma pack(pop)
#endif

    template <typename T>
    signed_type<T> signed_type<T>::invalid_signed_type =
        signed_type<T>();

}}}

namespace hpx { namespace traits
{
    template <typename T>
    struct is_bitwise_serializable<
            hpx::components::security::signed_type<T> >
       : std::true_type
    {};
}}

#endif

#endif
