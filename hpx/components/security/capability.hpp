//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_SECURITY_SERVER_CAPABILITY_HPP
#define HPX_COMPONENTS_SECURITY_SERVER_CAPABILITY_HPP

#include <boost/array.hpp>
#include <boost/io/ios_state.hpp>
#include <boost/serialization/bitset.hpp>
#include <boost/serialization/serialization.hpp>

#include <climits>

namespace hpx { namespace components { namespace security
{
#if defined(_MSC_VER)
#  pragma pack(push, 1)
#endif

    namespace traits
    {
        template <typename Enable = void>
        struct capability
        {
            static std::size_t const size = 8;
            static std::size_t const array_size = (size + CHAR_BIT - 1) / CHAR_BIT;

            enum capabilities
            {
                capability_certificate_authority = 0
            };
        };
    }

    namespace detail
    {
        static unsigned char const array_bits[] =
        {
            0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80
        };
    }

    class capability
    {
    public:
        capability()
        {
            std::fill(bits_.begin(), bits_.end(), 0);
        }

        capability(unsigned long bits)
        {
            std::fill(bits_.begin(), bits_.end(), 0);

            unsigned long mask = 0x1ul;
            for (std::size_t i = 0; i != security::traits::capability<>::size; ++i)
            {
                if (bits & mask)
                    set(i);
                mask <<= 1;
            }
        }

        void set(std::size_t position, bool value = true)
        {
            BOOST_ASSERT(position < security::traits::capability<>::size);

            if (value)
                bits_[position / CHAR_BIT] |= detail::array_bits[position % CHAR_BIT];
            else
                bits_[position / CHAR_BIT] &= ~detail::array_bits[position % CHAR_BIT];
        }

        friend std::ostream & operator<<(std::ostream & os,
                                            capability const & capability)
        {
            boost::io::ios_flags_saver ifs(os);
            os << "<capability \"";

            for (std::size_t i = 0; i != security::traits::capability<>::array_size; ++i)
            {
                os << std::hex
                    << std::nouppercase
                    << std::setfill('0')
                    << std::setw(2)
                    << static_cast<unsigned int>(capability.bits_[i]);
            }

            os << "\">";
            return os;
        }

        unsigned char const* begin() const
        {
            return reinterpret_cast<unsigned char const*>(this);
        }
        unsigned char const* end() const
        {
            return reinterpret_cast<unsigned char const*>(this) + size();
        }

        BOOST_CONSTEXPR static std::size_t size()
        {
            return sizeof(capability);
        }

    private:
        friend class boost::serialization::access;

        template <typename Archive>
        void serialize(Archive & ar, const unsigned int)
        {
            ar & bits_;
        }

        boost::array<
            unsigned char, security::traits::capability<>::array_size
        > bits_;
    };

#if defined(_MSC_VER)
#  pragma pack(pop)
#endif
}}}

namespace boost { namespace serialization
{
    template <>
    struct is_bitwise_serializable<
            hpx::components::security::capability>
       : mpl::true_
    {};
}}

#endif
