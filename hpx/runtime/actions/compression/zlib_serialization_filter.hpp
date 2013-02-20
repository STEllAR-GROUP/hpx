//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_ACTION_ZLIB_SERIALIZATION_FILTER_FEB_15_2013_0935AM)
#define HPX_ACTION_ZLIB_SERIALIZATION_FILTER_FEB_15_2013_0935AM

#include <hpx/hpx_fwd.hpp>

#if defined(HPX_HAVE_ZLIB_COMPRESSION)
#include <hpx/config/forceinline.hpp>
#include <hpx/traits/action_serialization_filter.hpp>
#include <hpx/runtime/actions/guid_initialization.hpp>
#include <hpx/util/binary_filter.hpp>
#include <hpx/util/detail/serialization_registration.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>
#include <boost/iostreams/filter/zlib.hpp>

#include <memory>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    namespace detail
    {
        class zlib_compdecomp
          : public boost::iostreams::detail::zlib_base,
            public boost::iostreams::detail::zlib_allocator<std::allocator<char> >
        {
            typedef
                boost::iostreams::detail::zlib_allocator<std::allocator<char> >
            allocator_type;

        public:
            zlib_compdecomp(bool compress = false,
                boost::iostreams::zlib_params const& params =
                    boost::iostreams::zlib::default_compression);
            ~zlib_compdecomp();

            bool save(char const*& src_begin, char const* src_end,
                char*& dest_begin, char* dest_end, bool flush = false);
            bool load(char const*& begin_in, char const* end_in,
                char*& begin_out, char* end_out);

            void close();

            bool eof() const { return eof_; }

        private:
            bool compress_;
            bool eof_;
        };
    }

    struct HPX_EXPORT zlib_serialization_filter : public util::binary_filter
    {
        zlib_serialization_filter(bool compress = false,
                binary_filter::mode m = binary_filter::favor_speed)
          : compdecomp_(compress), current_(0)
        {}
        ~zlib_serialization_filter();

        std::size_t load(void* dst, std::size_t dst_count,
            void const* src, std::size_t src_count);
        std::size_t save(void* dst, std::size_t dst_count,
            void const* src, std::size_t src_count);
        std::size_t flush(void* dst, std::size_t dst_count);

        void set_max_compression_length(std::size_t size);
        void init_decompression_data(char const* buffer, std::size_t size,
            std::size_t decompressed_size);

        /// serialization support
        static void register_base();

    protected:
        std::size_t load_impl(void* dst, std::size_t dst_count,
            void const* src, std::size_t src_count);

    private:
        // serialization support
        friend class boost::serialization::access;

        template <typename Archive>
        BOOST_FORCEINLINE void serialize(Archive& ar, const unsigned int) {}

        detail::zlib_compdecomp compdecomp_;
        std::vector<char> buffer_;
        std::size_t current_;
    };
}}

#include <hpx/config/warnings_suffix.hpp>

HPX_SERIALIZATION_REGISTER_TYPE_DECLARATION(hpx::actions::zlib_serialization_filter);

///////////////////////////////////////////////////////////////////////////////
#define HPX_ACTION_USES_FAST_ZLIB_COMPRESSION(action)                         \
    namespace hpx { namespace traits                                          \
    {                                                                         \
        template <>                                                           \
        struct action_serialization_filter<action>                            \
        {                                                                     \
            /* Note that the caller is responsible for deleting the filter */ \
            /* instance returned from this function */                        \
            static util::binary_filter* call()                                \
            {                                                                 \
                return new hpx::actions::zlib_serialization_filter(true,      \
                    util::binary_filter::favor_speed);                        \
            }                                                                 \
        };                                                                    \
    }}                                                                        \
/**/

#define HPX_ACTION_USES_ZLIB_COMPRESSION(action)                              \
    HPX_ACTION_USES_FAST_ZLIB_COMPRESSION(action)                             \
/**/

#else

#define HPX_ACTION_USES_ZLIB_COMPRESSION(action)
#define HPX_ACTION_USES_FAST_ZLIB_COMPRESSION(action)

#endif
#endif
