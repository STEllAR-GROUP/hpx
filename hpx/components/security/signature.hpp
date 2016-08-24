//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_SECURITY_SERVER_SIGNATURE_HPP
#define HPX_COMPONENTS_SECURITY_SERVER_SIGNATURE_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_SECURITY)

#include <hpx/runtime/serialization/array.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/traits/is_bitwise_serializable.hpp>

#include <boost/array.hpp>
#include <boost/io/ios_state.hpp>

#include <cstddef>

#include <sodium.h>

namespace hpx { namespace components { namespace security
{
#if defined(HPX_MSVC)
#  pragma pack(push, 1)
#endif

    class signature
    {
    public:
        signature()
        {
            std::fill(bytes_.begin(), bytes_.end(), 0);
        }

        friend std::ostream & operator<<(std::ostream & os,
                                         signature const & signature)
        {
            boost::io::ios_flags_saver ifs(os);

            os << "<signature \"";

            for (std::size_t i = 0; i < crypto_sign_BYTES; ++i)
            {
                os << std::hex
                   << std::nouppercase
                   << std::setfill('0')
                   << std::setw(2)
                   << static_cast<unsigned int>(signature.bytes_[i]);
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
            unsigned char, crypto_sign_BYTES
        > bytes_;
    };

#if defined(HPX_MSVC)
#  pragma pack(pop)
#endif
}}}

HPX_IS_BITWISE_SERIALIZABLE(hpx::components::security::signature)

#endif

#endif
