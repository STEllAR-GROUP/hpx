//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_SECURITY_SERVER_CAPABILITY_HPP
#define HPX_COMPONENTS_SECURITY_SERVER_CAPABILITY_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_SECURITY)

#include <hpx/traits/is_bitwise_serializable.hpp>

#include <boost/array.hpp>
#include <boost/io/ios_state.hpp>
#include <boost/mpl/bool.hpp>

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

            // NOTE: Each second capability is assumed to be a delegation capability
            enum capabilities
            {
                capability_certificate_authority = 0,
                capability_certificate_authority_delegation = 1,
                capability_create_component = 2,
                capability_create_component_delegation = 3,
                capability_const = 4,
                capability_const_delegation = 5,
                capability_non_const = 6,
                capability_non_const_delegation = 7
            };

            static boost::uint64_t const root_certificate_authority_capability =
                (1ull << capability_certificate_authority) +
                (1ull << capability_create_component_delegation) +
                (1ull << capability_const_delegation) +
                (1ull << capability_non_const_delegation);
        };
    }

    class capability
    {
    public:
        capability()
        {
            std::fill(bits_.begin(), bits_.end(), 0);
        }

        capability(boost::uint64_t bits)
        {
            std::fill(bits_.begin(), bits_.end(), 0);
            for (std::size_t i = 0;
                 i != security::traits::capability<>::size;
                 ++i)
            {
                set(i, (bits & (1ull << i)) != 0);
            }
        }

        void set(std::size_t position, bool value = true)
        {
            HPX_ASSERT(position < security::traits::capability<>::size);

            if (value)
            {
                bits_[position / CHAR_BIT] |= (1ull << (position % CHAR_BIT));
            }
            else
            {
                bits_[position / CHAR_BIT] &= ~(1ull << (position % CHAR_BIT));
            }
        }

        bool test(std::size_t position) const
        {
            HPX_ASSERT(position < security::traits::capability<>::size);

            return (bits_[position / CHAR_BIT] & (1ull << (position % CHAR_BIT))) != 0;
        }

        bool verify(capability const & sender) const
        {
            // Verify sender has the capabilities to this

            for (std::size_t i = 0; i < traits::capability<>::size; ++i)
            {
                if (test(i) == true && sender.test(i) == false)
                {
                    return false;
                }
            }

            return true;
        }

        bool verify_delegation(capability const & subject) const
        {
            // Verify that this can delegate the capabilities requested by subject

            for (std::size_t i = 0; i < traits::capability<>::size; i += 2)
            {
                if (subject.test(i) == true && test(i + 1) == false)
                {
                    return false;
                }
            }

            return true;
        }

        friend std::ostream & operator<<(std::ostream & os,
                                            capability const & capability)
        {
            os << "<capability \"";

            for (std::size_t i = 0;
                 i != ((traits::capability<>::size + CHAR_BIT - 1) / CHAR_BIT);
                 ++i)
            {
                for (std::size_t j = CHAR_BIT; j != 0; --j)
                {
                    os << ((capability.bits_[i] & (1ull << (j - 1))) == 0 ? "0"
                                                                          : "1");
                }
            }

            return os << "\">";
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
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive & ar, const unsigned int)
        {
            ar & bits_;
        }

        boost::array<
            unsigned char
          , (traits::capability<>::size + CHAR_BIT - 1) / CHAR_BIT
        > bits_;
    };

#if defined(_MSC_VER)
#  pragma pack(pop)
#endif
}}}

namespace hpx { namespace traits
{
    template <>
    struct is_bitwise_serializable<
            hpx::components::security::capability>
       : boost::mpl::true_
    {};
}}

#endif

#endif
