//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_FUNCTION_DETAILSERIALIZATION_REGISTRATION_HPP
#define HPX_FUNCTION_DETAILSERIALIZATION_REGISTRATION_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/detail/pp_strip_parens.hpp>

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/void_cast.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/extended_type_info.hpp>
#include <boost/serialization/extended_type_info_typeid.hpp>
#include <boost/uuid/sha1.hpp>
#include <boost/cstdint.hpp>
#include <boost/assert.hpp>

#include <typeinfo>

namespace hpx { namespace util { namespace detail
{
    template <typename T>
    struct guid_initializer_helper
    {
        typedef boost::archive::detail::extra_detail::guid_initializer<T> base_type;
        base_type guid_init;

        guid_initializer_helper()
            : guid_init()
        {
            T::register_base();
        }

        void initialize() const {}

        guid_initializer_helper const & export_guid() const {
            guid_init.export_guid();
            return *this;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    inline char
    to_digit(int number)
    {
        char number_tmp = static_cast<char>(number);
        if (number >= 0 && number <= 9)
        {
            return static_cast<char>(number_tmp + '0');
        }
        return static_cast<char>(number_tmp - 10 + 'A');
    }

#if defined(BOOST_DISABLE_ASSERTS)
    inline void convert_byte(boost::uint8_t b, char*& buffer, char const* /*end*/)
#else
    inline void convert_byte(boost::uint8_t b, char*& buffer, char const* end)
#endif
    {
        BOOST_ASSERT(buffer < end-1);
        *buffer++ = to_digit((b & 0xF0) >> 4);
        *buffer++ = to_digit(b & 0x0F);
    }

    inline void
    convert_unsigned(boost::uint32_t ui, char*& buffer, char const* end)
    {
        convert_byte(static_cast<boost::uint8_t>(ui / 0x01000000), buffer, end);
        convert_byte(static_cast<boost::uint8_t>(ui / 0x010000 % 0x0100), buffer, end);
        convert_byte(static_cast<boost::uint8_t>(ui / 0x0100 % 0x0100), buffer, end);
        convert_byte(static_cast<boost::uint8_t>(ui % 0x0100), buffer, end);
    }

    template <typename T>
    char const* type_hash_string()
    {
        /// FIXME: this is not portable across different compilers
        char const* name = typeid(T).name();

        // create a sha1 hash from the string returned by typeid::name
        boost::uuids::detail::sha1 hash;
        hash.process_block(name, name + std::strlen(name));

        boost::uint32_t digest[5];
        hash.get_digest(digest);

        // 5 times the number of digits generated from a 32 bit unsigned
        static char buf[5 * 8 + 1] = { '\0' };

        // stringify hash
        char* buffer = buf;
        for (int i = 0; i < 5; ++i)
            convert_unsigned(digest[i], buffer, &buf[sizeof(buf)]);

        BOOST_ASSERT(buffer == &buf[sizeof(buf)-1]);
        *buffer = '\0';

        return buf;
    }

    ///////////////////////////////////////////////////////////////////////////
    // return a unique string representing the type T
    template <typename T>
    char const* type_hash()
    {
        static char const* name = type_hash_string<T>();
        return name;
    }
}}}

#define HPX_SERIALIZATION_REGISTER_TEMPLATE(TEMPLATE, TYPE)                     \
namespace boost {                                                               \
    namespace serialization {                                                   \
        HPX_UTIL_STRIP(TEMPLATE)                                                \
        struct guid_defined<HPX_UTIL_STRIP(TYPE)> : mpl::true_ {};              \
                                                                                \
        namespace ext {                                                         \
            HPX_UTIL_STRIP(TEMPLATE)                                            \
            struct guid_impl<HPX_UTIL_STRIP(TYPE)>                              \
            {                                                                   \
                static inline const char * call()                               \
                {                                                               \
                    return                                                      \
                        hpx::util::detail::type_hash<HPX_UTIL_STRIP(TYPE)>();   \
                }                                                               \
            };                                                                  \
        }                                                                       \
    }                                                                           \
    namespace archive { namespace detail { namespace extra_detail {             \
        HPX_UTIL_STRIP(TEMPLATE)                                                \
        struct init_guid<HPX_UTIL_STRIP(TYPE)>                                  \
        {                                                                       \
            static                                                              \
                hpx::util::detail::guid_initializer_helper<                     \
                    HPX_UTIL_STRIP(TYPE)                                        \
                > const &                                                       \
                g;                                                              \
        };                                                                      \
                                                                                \
        HPX_UTIL_STRIP(TEMPLATE)                                                \
        hpx::util::detail::guid_initializer_helper<HPX_UTIL_STRIP(TYPE)> const &\
        init_guid<HPX_UTIL_STRIP(TYPE)>::g                                      \
            = ::boost::serialization::singleton<                                \
                hpx::util::detail::guid_initializer_helper<HPX_UTIL_STRIP(TYPE)>\
            >::get_mutable_instance().export_guid();                            \
    }}}                                                                         \
}                                                                               \
/**/

#define HPX_SERIALIZATION_REGISTER_TEMPLATE_ACTION(TEMPLATE, ACTION)            \
    HPX_SERIALIZATION_REGISTER_TEMPLATE(TEMPLATE,                               \
        (hpx::actions::transfer_action<HPX_UTIL_STRIP(ACTION)>))                \
/**/

#endif

