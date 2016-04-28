//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_ACTION_SNAPPY_SERIALIZATION_FILTER_REGISTRATION_APR_28_2016_1008AM)
#define HPX_ACTION_SNAPPY_SERIALIZATION_FILTER_REGISTRATION_APR_28_2016_1008AM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_COMPRESSION_SNAPPY)

#include <hpx/traits/action_serialization_filter.hpp>

///////////////////////////////////////////////////////////////////////////////
#define HPX_ACTION_USES_SNAPPY_COMPRESSION(action)                            \
    namespace hpx { namespace traits                                          \
    {                                                                         \
        template <>                                                           \
        struct action_serialization_filter<action>                            \
        {                                                                     \
            /* Note that the caller is responsible for deleting the filter */ \
            /* instance returned from this function */                        \
            static serialization::binary_filter* call(                        \
                    parcelset::parcel const& p)                               \
            {                                                                 \
                return hpx::create_binary_filter(                             \
                    "snappy_serialization_filter", true);                     \
            }                                                                 \
        };                                                                    \
    }}                                                                        \
/**/

#else

#define HPX_ACTION_USES_SNAPPY_COMPRESSION(action)

#endif
#endif
