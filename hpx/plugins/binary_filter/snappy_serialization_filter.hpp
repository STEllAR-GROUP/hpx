//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_ACTION_SNAPPY_SERIALIZATION_FILTER_FEB_21_2013_0203PM)
#define HPX_ACTION_SNAPPY_SERIALIZATION_FILTER_FEB_21_2013_0203PM

#include <hpx/hpx_fwd.hpp>

#if defined(HPX_HAVE_COMPRESSION_SNAPPY)
#include <hpx/config/forceinline.hpp>
#include <hpx/traits/action_serialization_filter.hpp>
#include <hpx/runtime/serialization/binary_filter.hpp>

#include <boost/iostreams/filter/zlib.hpp>

#include <memory>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace plugins { namespace compression
{
    struct HPX_LIBRARY_EXPORT snappy_serialization_filter
      : public serialization::binary_filter
    {
        snappy_serialization_filter(bool compress = false,
                serialization::binary_filter* next_filter = 0)
          : current_(0), compress_(compress)
        {}

        void load(void* dst, std::size_t dst_count);
        void save(void const* src, std::size_t src_count);
        bool flush(void* dst, std::size_t dst_count, std::size_t& written);

        void set_max_length(std::size_t size);
        std::size_t init_data(char const* buffer,
            std::size_t size, std::size_t buffer_size);

    private:
        // serialization support
        friend class hpx::serialization::access;

        template <typename Archive>
        BOOST_FORCEINLINE void serialize(Archive& ar, const unsigned int) {}

        HPX_SERIALIZATION_POLYMORPHIC(snappy_serialization_filter);

        std::vector<char> buffer_;
        std::size_t current_;
        bool compress_;
    };
}}}

#include <hpx/config/warnings_suffix.hpp>

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
