#ifndef PORTABLE_BINARY_OARCHIVE_HPP
#define PORTABLE_BINARY_OARCHIVE_HPP

#include <boost/version.hpp>
#include <hpx/config.hpp>

#if !defined(HPX_USE_PORTABLE_ARCHIVES) || HPX_USE_PORTABLE_ARCHIVES == 0
#include <boost/archive/binary_oarchive.hpp>

namespace hox { namespace util
{
    typedef boost::archive::binary_oarchive portable_binary_oarchive;
}}

#else

// MS compatible compilers support #pragma once
#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

#if defined(_MSC_VER)
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

/////////1/////////2/////////3/////////4/////////5/////////6/////////7/////////8
// portable_binary_oarchive.hpp

// (C) Copyright 2002 Robert Ramey - http://www.rrsd.com .
// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

//  See http://www.boost.org for updates, documentation, and revision history.

#include <ostream>
#include <boost/serialization/string.hpp>
#include <boost/archive/archive_exception.hpp>

#if !defined(BOOST_WINDOWS)
  #pragma GCC visibility push(default)
#endif

#include <boost/archive/basic_binary_oprimitive.hpp>

#if !defined(BOOST_WINDOWS)
  #pragma GCC visibility pop
#endif

#include <boost/archive/detail/common_oarchive.hpp>
#include <boost/archive/detail/register_archive.hpp>
#if BOOST_VERSION >= 104400
#include <boost/serialization/item_version_type.hpp>
#endif

#include <hpx/config.hpp>
#include <hpx/util/portable_binary_archive.hpp>

namespace hpx { namespace util
{

/////////1/////////2/////////3/////////4/////////5/////////6/////////7/////////8
// exception to be thrown if integer read from archive doesn't fit
// variable being loaded
class HPX_ALWAYS_EXPORT portable_binary_oarchive_exception :
    public virtual boost::archive::archive_exception
{
public:
    enum exception_code {
        invalid_flags
    };
    portable_binary_oarchive_exception(exception_code c = invalid_flags)
      : boost::archive::archive_exception(static_cast<boost::archive::archive_exception::exception_code>(c))
    {}
    virtual const char *what() const throw()
    {
        const char *msg = "programmer error";
        switch(code){
        case invalid_flags:
            msg = "cannot be both big and little endian";
        default:
            boost::archive::archive_exception::what();
        }
        return msg;
    }
};

/////////1/////////2/////////3/////////4/////////5/////////6/////////7/////////8
// "Portable" output binary archive.  This is a variation of the native binary
// archive. it addresses integer size and endianness so that binary archives can
// be passed across systems. Note:floating point types not addressed here

#if defined(BOOST_MSVC)
#define HPX_SERIALIZATION_EXPORT
#else
#define HPX_SERIALIZATION_EXPORT HPX_ALWAYS_EXPORT
#endif

class HPX_SERIALIZATION_EXPORT portable_binary_oarchive :
    public boost::archive::basic_binary_oprimitive<
        portable_binary_oarchive,
        std::ostream::char_type,
        std::ostream::traits_type
    >,
    public boost::archive::detail::common_oarchive<
        portable_binary_oarchive
    >
{
    typedef boost::archive::basic_binary_oprimitive<
        portable_binary_oarchive,
        std::ostream::char_type,
        std::ostream::traits_type
    > primitive_base_t;
    typedef boost::archive::detail::common_oarchive<
        portable_binary_oarchive
    > archive_base_t;
#ifndef BOOST_NO_MEMBER_TEMPLATE_FRIENDS
public:
#else
    friend archive_base_t;
    friend primitive_base_t; // since with override save below
    friend class boost::archive::detail::interface_oarchive<
        portable_binary_oarchive
    >;
    friend class boost::archive::save_access;
protected:
#endif
    unsigned int m_flags;
    HPX_SERIALIZATION_EXPORT void
    save_impl(const boost::intmax_t l, const char maxsize);
    // add base class to the places considered when matching
    // save function to a specific set of arguments.  Note, this didn't
    // work on my MSVC 7.0 system so we use the sure-fire method below
    // using archive_base_t::save;

    // default fall through for any types not specified here
    template<class T>
    HPX_SERIALIZATION_EXPORT void save(const T & val) {
        boost::intmax_t t = static_cast<boost::intmax_t>(val);
        save_impl(t, sizeof(T));
    }
    HPX_SERIALIZATION_EXPORT void save(const std::string & t) {
        this->primitive_base_t::save(t);
    }
#if BOOST_VERSION >= 104400
    HPX_SERIALIZATION_EXPORT void save(const boost::archive::class_id_type & t) {
        /*boost::int16_t*/boost::intmax_t l = t;
        save_impl(l, sizeof(boost::int16_t));
    }
    HPX_SERIALIZATION_EXPORT void save(const boost::archive::object_id_type & t) {
        /*boost::uint32_t*/boost::intmax_t l = t;
        save_impl(l, sizeof(boost::uint32_t));
    }
    HPX_SERIALIZATION_EXPORT void save(const boost::archive::tracking_type & t) {
        bool l = t;
        this->primitive_base_t::save(l);
    }
    HPX_SERIALIZATION_EXPORT void save(const boost::archive::version_type & t) {
        /*boost::uint32_t*/boost::intmax_t l = t;
        save_impl(l, sizeof(boost::uint32_t));
    }
    HPX_SERIALIZATION_EXPORT void save(const boost::archive::library_version_type & t) {
        /*boost::uint16_t*/boost::intmax_t l = t;
        save_impl(l, sizeof(boost::uint16_t));
    }
    HPX_SERIALIZATION_EXPORT void save(const boost::serialization::item_version_type & t) {
        boost::intmax_t l = t;
        save_impl(l, sizeof(boost::intmax_t));
    }
#endif
#ifndef BOOST_NO_STD_WSTRING
    HPX_SERIALIZATION_EXPORT void save(const std::wstring & t) {
        this->primitive_base_t::save(t);
    }
#endif
    HPX_SERIALIZATION_EXPORT void save(const float & t) {
        this->primitive_base_t::save(t);
        // floats not supported
        //BOOST_STATIC_ASSERT(false);
    }
    HPX_SERIALIZATION_EXPORT void save(const double & t) {
        this->primitive_base_t::save(t);
        // doubles not supported
        //BOOST_STATIC_ASSERT(false);
    }
    HPX_SERIALIZATION_EXPORT void save(const char & t) {
        this->primitive_base_t::save(t);
    }
    HPX_SERIALIZATION_EXPORT void save(const unsigned char & t) {
        this->primitive_base_t::save(t);
    }

    // default processing - kick back to base class.  Note the
    // extra stuff to get it passed borland compilers
    typedef boost::archive::detail::common_oarchive<portable_binary_oarchive>
        detail_common_oarchive;

    template<class T>
    void save_override(T & t, BOOST_PFTO int) {
        this->detail_common_oarchive::save_override(t, 0);
    }
    // explicitly convert to char * to avoid compile ambiguities
    void save_override(const boost::archive::class_name_type & t, int) {
        const std::string s(t);
        *this << s;
    }

    // binary files don't include the optional information
    void save_override(const boost::archive::class_id_optional_type&, int) {}

    HPX_SERIALIZATION_EXPORT void
    init(unsigned int flags);

public:
    portable_binary_oarchive(std::ostream & os, unsigned flags = 0)
      : primitive_base_t(
            * os.rdbuf(),
            0 != (flags & boost::archive::no_codecvt)
        ),
        archive_base_t(flags),
        m_flags(flags & (endian_big | endian_little))
    {
        init(flags);
    }

    portable_binary_oarchive(
            std::basic_streambuf<
                std::ostream::char_type,
                std::ostream::traits_type
            > & bsb,
            unsigned int flags)
      : primitive_base_t(
            bsb,
            0 != (flags & boost::archive::no_codecvt)
        ),
        archive_base_t(flags),
        m_flags(0)
    {
        init(flags);
    }
};

#undef HPX_SERIALIZATION_EXPORT

}}

// required by export in boost version > 1.34
#ifdef BOOST_SERIALIZATION_REGISTER_ARCHIVE
    BOOST_SERIALIZATION_REGISTER_ARCHIVE(hpx::util::portable_binary_oarchive)
#endif
#ifdef BOOST_SERIALIZATION_USE_ARRAY_OPTIMIZATION
    BOOST_SERIALIZATION_USE_ARRAY_OPTIMIZATION(hpx::util::portable_binary_oarchive)
#endif

#if defined(_MSC_VER)
#pragma warning( pop )
#endif

#endif // PORTABLE_BINARY_OARCHIVE_HPP

#endif // HPX_USE_PORTABLE_ARCHIVES == 0

