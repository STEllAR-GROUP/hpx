/////////1/////////2/////////3/////////4/////////5/////////6/////////7/////////8
// portable_binary_iarchive.cpp

// (C) Copyright 2002 Robert Ramey - http://www.rrsd.com .
// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

//  See http://www.boost.org for updates, documentation, and revision history.

#include <hpx/config.hpp>
#include <boost/version.hpp>
#include <boost/config.hpp>

#if BOOST_VERSION >= 103700 && HPX_USE_PORTABLE_ARCHIVES != 0

// export the defined functions
#define BOOST_ARCHIVE_SOURCE

// this hack is needed to properly compile this shared library, allowing to
// export the symbols and auto link with the serialization
#if !defined(BOOST_ALL_NO_LIB) && !defined(BOOST_SERIALIZATION_NO_LIB)
// Set the name of our library, this will get undef'ed by auto_link.hpp
// once it's done with it:
#define BOOST_LIB_NAME boost_serialization

// If we're importing code from a dll, then tell auto_link.hpp about it:
#if defined(BOOST_ALL_DYN_LINK) || defined(BOOST_SERIALIZATION_DYN_LINK)
#  define BOOST_DYN_LINK
#endif

// And include the header that does the work:
#include <boost/config/auto_link.hpp>
#endif  // auto-linking disabled

#include <istream>
#include <string>
//#include <cstring> // memcpy

#include <boost/detail/endian.hpp>
#include <boost/throw_exception.hpp>
#include <boost/archive/archive_exception.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>

#if defined(BOOST_MSVC)
#  include <intrin.h>
#  pragma intrinsic(memcpy)
#  pragma intrinsic(memset)
#endif

namespace hpx { namespace util
{

void portable_binary_iarchive::load_impl(boost::int64_t& l, char maxsize)
{
    l = 0;

    char size;
    this->primitive_base_t::load(size);
    if (0 == size)
        return;

    if (size > maxsize) {
        BOOST_THROW_EXCEPTION(portable_binary_iarchive_exception());
    }

    bool negative;
    this->primitive_base_t::load(negative);

    char* cptr = reinterpret_cast<char *>(&l);
#ifdef BOOST_BIG_ENDIAN
    cptr += (sizeof(boost::int64_t) - size);
#endif
    this->primitive_base_t::load_binary(cptr, static_cast<std::size_t>(size));

#ifdef BOOST_BIG_ENDIAN
    if(m_flags & endian_little)
          reverse_bytes(size, cptr);
#else
    if(m_flags & endian_big)
          reverse_bytes(size, cptr);
#endif

    if(negative)
        l = -l;
}

void portable_binary_iarchive::load_override(
    boost::archive::class_name_type& t, int)
{
    std::string cn;
    cn.reserve(BOOST_SERIALIZATION_MAX_KEY_SIZE);
    load_override(cn, 0);
    if(cn.size() > (BOOST_SERIALIZATION_MAX_KEY_SIZE - 1)) {
        BOOST_THROW_EXCEPTION(
            boost::archive::archive_exception(
                boost::archive::archive_exception::invalid_class_name));
    }

    std::memcpy(t, cn.data(), cn.size());
    // borland tweak
    t.t[cn.size()] = '\0';
}

#if defined(__GNUG__) && !defined(__INTEL_COMPILER)
#if defined(HPX_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#pragma GCC diagnostic push
#endif
#pragma GCC diagnostic ignored "-Wconversion"
#endif

void portable_binary_iarchive::init(unsigned int flags)
{
    if (!(flags & boost::archive::no_header))
    {
        // read signature in an archive version independent manner
        std::string file_signature;
        *this >> file_signature;
        if (file_signature != boost::archive::BOOST_ARCHIVE_SIGNATURE()) {
            BOOST_THROW_EXCEPTION(
                boost::archive::archive_exception(
                    boost::archive::archive_exception::invalid_signature));
        }

        // make sure the version of the reading archive library can
        // support the format of the archive being read
        boost::archive::version_type input_library_version(0);
        *this >> input_library_version;

        // extra little .t is to get around borland quirk
        if (boost::archive::BOOST_ARCHIVE_VERSION() < input_library_version) {
            BOOST_THROW_EXCEPTION(
                boost::archive::archive_exception(
                    boost::archive::archive_exception::unsupported_version));
        }

#if BOOST_WORKAROUND(__MWERKS__, BOOST_TESTED_AT(0x3205))
        set_library_version(input_library_version);
        boost::archive::detail::basic_iarchive::set_library_version(
            input_library_version);
#endif
    }

    unsigned char x;
    load(x);
    m_flags = static_cast<unsigned int>(x << CHAR_BIT);

    // handle filter and compression in the archive separately
    bool has_filter = false;
    *this >> has_filter;

    if (has_filter) {
        util::binary_filter* filter = 0;
        *this >> filter;
        if (m_flags & enable_compression)
            this->set_filter(filter);
    }
}

#if defined(__GNUG__) && !defined(__INTEL_COMPILER)
#if defined(HPX_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#pragma GCC diagnostic pop
#endif
#endif

}}

// explicitly instantiate for this type
#if BOOST_VERSION < 104000
#include <boost/archive/impl/archive_pointer_iserializer.ipp>

namespace boost {
namespace archive {

    template class HPX_ALWAYS_EXPORT
        detail::archive_pointer_iserializer<hpx::util::portable_binary_iarchive>;

} // namespace archive
} // namespace boost
#else
#include <boost/archive/detail/archive_serializer_map.hpp>
#include <boost/archive/impl/archive_serializer_map.ipp>

namespace boost {
namespace archive {

    template class HPX_ALWAYS_EXPORT
        detail::archive_serializer_map<hpx::util::portable_binary_iarchive>;

} // namespace archive
} // namespace boost

#endif

// explicitly instantiate for this type of stream
#include <hpx/util/basic_binary_iprimitive_impl.hpp>

namespace hpx { namespace util
{
    template class basic_binary_iprimitive<
        hpx::util::portable_binary_iarchive
    >;
}}

#endif // BOOST_VERSION >= 103700 && HPX_USE_PORTABLE_ARCHIVES != 0
