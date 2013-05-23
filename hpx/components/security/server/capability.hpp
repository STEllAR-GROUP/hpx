//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_SECURITY_SERVER_CAPABILITY_HPP
#define HPX_COMPONENTS_SECURITY_SERVER_CAPABILITY_HPP

#include <boost/serialization/bitset.hpp>
#include <boost/serialization/serialization.hpp>

#include <bitset>
#include <string>

namespace hpx { namespace components { namespace security
{
    namespace traits
    {
        template <typename Enable = void>
        struct capability
        {
            static std::size_t const size = 8;

            enum capabilities
            {
                capability_certificate_authority = 0
            };
        };
    }

    namespace server
    {
        class capability
        {
        public:
            capability()
            {
            };

            capability(unsigned long bits)
              : bits_(bits)
            {
            }

            capability(std::string const & bits)
              : bits_(bits)
            {
            }

            void
            set(std::size_t position, bool value = true)
            {
                bits_.set(position, value);
            }

        private:
            friend class boost::serialization::access;

            template <typename Archive>
            void serialize(Archive & ar, const unsigned int)
            {
                ar & bits_;
            }

            std::bitset<traits::capability<>::size> bits_;
        };
    }
}}}

#endif
