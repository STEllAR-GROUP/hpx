/////////1/////////2/////////3/////////4/////////5/////////6/////////7/////////8
// portable_binary_oarchive.cpp

// (C) Copyright 2002-7 Robert Ramey - http://www.rrsd.com .
// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

//  See http://www.boost.org for updates, documentation, and revision history.

#include <boost/version.hpp>
#include <hpx/config.hpp>

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

#include <ostream>

#include <boost/detail/endian.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

namespace hpx { namespace util
{

void portable_binary_oarchive::save_impl(
    const boost::intmax_t l, const char maxsize)
{
    char size = 0;
    if (l == 0) {
        this->primitive_base_t::save(size);
        return;
    }

    boost::intmax_t ll;
    bool negative = (l < 0);
    if (negative)
        ll = -l;
    else
        ll = l;

    do {
        ll >>= CHAR_BIT;
        ++size;
    } while(ll != 0);

    this->primitive_base_t::save(static_cast<char>(negative ? -size : size));

    if(negative)
        ll = -l;
    else
        ll = l;

    char* cptr = reinterpret_cast<char *>(& ll);
#ifdef BOOST_BIG_ENDIAN
    cptr += (sizeof(boost::intmax_t) - size);
    if(m_flags & endian_little)
        reverse_bytes(size, cptr);
#else
    if(m_flags & endian_big)
        reverse_bytes(size, cptr);
#endif
    this->primitive_base_t::save_binary(cptr, static_cast<std::size_t>(size));
}

#if defined(__GNUG__) && !defined(__INTEL_COMPILER)
#   if defined(HPX_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#       pragma GCC diagnostic push
#   endif
#   pragma GCC diagnostic ignored "-Wconversion"
#endif
void portable_binary_oarchive::init(unsigned int flags)
{
    if (m_flags == (endian_big | endian_little)) {
        BOOST_THROW_EXCEPTION(
            portable_binary_oarchive_exception());
    }

    if (0 == (flags & boost::archive::no_header)) {
        // write signature in an archive version independent manner
        const std::string file_signature(
            boost::archive::BOOST_ARCHIVE_SIGNATURE());
        *this << file_signature;

        // write library version
        const boost::archive::version_type v(
            boost::archive::BOOST_ARCHIVE_VERSION());
        *this << v;
    }
    save(static_cast<unsigned char>(m_flags >> CHAR_BIT));
}
#if defined(__GNUG__) && !defined(__INTEL_COMPILER)
#   if defined(HPX_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#       pragma GCC diagnostic pop
#   endif
#endif

}}

#if BOOST_VERSION < 104000
// explicitly instantiate for this type of stream
#include <boost/archive/impl/archive_pointer_oserializer.ipp>

namespace boost {
namespace archive {

template class HPX_ALWAYS_EXPORT
    detail::archive_pointer_oserializer<hpx::util::portable_binary_oarchive>;

} // namespace archive
} // namespace boost
#else
#include <boost/archive/detail/archive_serializer_map.hpp>
#include <boost/archive/impl/archive_serializer_map.ipp>

namespace boost {
namespace archive {

template class HPX_ALWAYS_EXPORT
    detail::archive_serializer_map<hpx::util::portable_binary_oarchive>;

} // namespace archive
} // namespace boost

#endif

// explicitly instantiate for this type of stream
#include <boost/archive/impl/basic_binary_oprimitive.ipp>

namespace boost {
namespace archive {

template class basic_binary_oprimitive<
    hpx::util::portable_binary_oarchive,
    std::ostream::char_type,
    std::ostream::traits_type
>;

} // namespace archive
} // namespace boost

#endif // BOOST_VERSION >= 103700 && HPX_USE_PORTABLE_ARCHIVES != 0
