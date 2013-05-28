//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_SECURITY_SERVER_HASH_HPP
#define HPX_COMPONENTS_SECURITY_SERVER_HASH_HPP

#include <boost/array.hpp>
#include <hpx/exception.hpp>
#include <sodium.h>

namespace hpx { namespace components { namespace security { namespace server
{
    namespace traits
    {
        template <typename Enable = void>
        struct hash
        {
            typedef crypto_generichash_state state_type;

            typedef boost::array<
                unsigned char, crypto_generichash_BYTES_MAX
            > final_type;

            static void
            init(state_type & state)
            {
                if (crypto_generichash_init(
                        &state, NULL, 0, final_type::static_size) != 0)
                {
                    HPX_THROW_EXCEPTION(
                        hpx::security_error
                      , "hash::init"
                      , "Failed to initialise hash state"
                    )
                }
            }

            static void
            update(
                state_type & state
              , unsigned char const * input
              , std::size_t input_length)
            {
                if (crypto_generichash_update(
                        &state, input, input_length) != 0)
                {
                    HPX_THROW_EXCEPTION(
                        hpx::security_error
                      , "hash::update"
                      , "Failed to update hash state"
                    )
                }
            }

            static final_type
            final(state_type & state)
            {
                final_type final;

                if (crypto_generichash_final(
                        &state
                      , final.c_array()
                      , final_type::static_size) != 0)
                {
                    HPX_THROW_EXCEPTION(
                        hpx::security_error
                      , "hash::final"
                      , "Failed to finalise hash state"
                    )
                }

                return final;
            }
        };
    }

    class hash
    {
    public:
        hash()
        {
            traits::hash<>::init(state_);
        }

        void
        update(unsigned char const * input, std::size_t input_length)
        {
            traits::hash<>::update(state_, input, input_length);
        }

        traits::hash<>::final_type
        final()
        {
            return traits::hash<>::final(state_);
        }

    private:
        traits::hash<>::state_type state_;
    };
}}}}

#endif
