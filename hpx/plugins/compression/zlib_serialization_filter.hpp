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
#include <boost/iostreams/filter/zlib.hpp>

#include <memory>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace plugins { namespace compression
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

    struct HPX_LIBRARY_EXPORT zlib_serialization_filter : public util::binary_filter
    {
        zlib_serialization_filter(bool compress = false)
          : compdecomp_(compress), current_(0)
        {
            // The zlib serialization filter does not support the mode 
            // 'favour_memorysize'.
            BOOST_ASSERT(m == binary_filter::favor_speed);
        }
        ~zlib_serialization_filter();

        void load(void* dst, std::size_t dst_count);
        void save(void const* src, std::size_t src_count);
        bool flush(void* dst, std::size_t dst_count, std::size_t& written);

        void set_max_compression_length(std::size_t size);
        std::size_t init_decompression_data(char const* buffer, 
            std::size_t size, std::size_t decompressed_size);

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
}}}

#include <hpx/config/warnings_suffix.hpp>

HPX_SERIALIZATION_REGISTER_TYPE_DECLARATION(
    hpx::plugins::compression::zlib_serialization_filter);

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
                using namespace hpx::plugins::compression;                    \
                return new zlib_serialization_filter(true);                   \
            }                                                                 \
        };                                                                    \
    }}                                                                        \
/**/

#endif
