//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_SECURITY_SERVER_HASH_HPP
#define HPX_COMPONENTS_SECURITY_SERVER_HASH_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_SECURITY)

#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/array.hpp>
#include <hpx/traits/is_bitwise_serializable.hpp>

#include <boost/array.hpp>
#include <boost/io/ios_state.hpp>

#include <sodium.h>

namespace hpx { namespace components { namespace security
{
#if defined(HPX_MSVC)
#  pragma pack(push, 1)
#endif

    class hash
    {
    public:
        hash()
        {
            crypto_hash(bytes_.c_array(), NULL, 0);
        }

        hash(unsigned char const * input, std::size_t input_length)
        {
            crypto_hash(bytes_.c_array(), input, input_length);
        }

        friend bool operator==(hash const & lhs, hash const & rhs)
        {
            // From libsodium's crypto_verify_16
            unsigned int different_bits = 0;

            for (std::size_t i = 0; i < crypto_hash_BYTES; ++i)
            {
                different_bits |= lhs.bytes_[i] ^ rhs.bytes_[i];
            }

            return ((1 & ((different_bits - 1) >> 8)) - 1) == 0;
        }

        friend bool operator!=(hash const & lhs, hash const & rhs)
        {
            return !(lhs == rhs);
        }

        friend std::ostream & operator<<(std::ostream & os,
                                         hash const & hash)
        {
            boost::io::ios_flags_saver ifs(os);

            os << "<hash \"";

            for (std::size_t i = 0; i < crypto_hash_BYTES; ++i)
            {
                os << std::hex
                   << std::nouppercase
                   << std::setfill('0')
                   << std::setw(2)
                   << static_cast<unsigned int>(hash.bytes_[i]);
            }

            return os << "\">";
        }

    private:
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive & ar, const unsigned int)
        {
            ar & bytes_;
        }

        boost::array<
            unsigned char, crypto_hash_BYTES
        > bytes_;
    };

#if defined(HPX_MSVC)
#  pragma pack(pop)
#endif
}}}

#endif

#endif
