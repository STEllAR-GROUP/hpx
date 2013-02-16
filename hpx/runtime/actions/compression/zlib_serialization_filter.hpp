//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_ACTION_ZLIB_SERIALIZATION_FILTER_FEB_15_2013_0935AM)
#define HPX_ACTION_ZLIB_SERIALIZATION_FILTER_FEB_15_2013_0935AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/config/forceinline.hpp>
#include <hpx/traits/action_serialization_filter.hpp>
#include <hpx/runtime/actions/guid_initialization.hpp>
#include <hpx/util/binary_filter.hpp>
#include <hpx/util/detail/serialization_registration.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>

#include <memory>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    struct HPX_EXPORT zlib_serialization_filter : public util::binary_filter
    {
        ~zlib_serialization_filter();

        std::size_t load(void* address, void const* src, std::size_t count);
        std::size_t save(void* dest, void const* address, std::size_t count);

        /// serialization support
        static void register_base();

    private:
        // serialization support
        friend class boost::serialization::access;

        template <typename Archive>
        BOOST_FORCEINLINE void serialize(Archive& ar, const unsigned int) {}
    };
}}

#include <hpx/config/warnings_suffix.hpp>

HPX_SERIALIZATION_REGISTER_TYPE_DECLARATION(hpx::actions::zlib_serialization_filter);

///////////////////////////////////////////////////////////////////////////////
#define HPX_ACTION_USES_ZLIB_COMPRESSION(action)                              \
    namespace hpx { namespace traits                                          \
    {                                                                         \
        template <>                                                           \
        struct action_serialization_filter<action>                            \
        {                                                                     \
            /* Note that the caller is responsible for deleting the filter */ \
            /* instance returned from this function */                        \
            static util::binary_filter* call()                                \
            {                                                                 \
                return new hpx::actions::zlib_serialization_filter;           \
            }                                                                 \
        };                                                                    \
    }}                                                                        \
/**/

#endif
