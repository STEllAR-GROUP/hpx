#ifndef PORTABLE_BINARY_OARCHIVE_HPP
#define PORTABLE_BINARY_OARCHIVE_HPP

#include <boost/version.hpp>
#include <hpx/config.hpp>

#if !defined(HPX_USE_PORTABLE_ARCHIVES) || HPX_USE_PORTABLE_ARCHIVES == 0
#include <boost/archive/binary_oarchive.hpp>

namespace hpx { namespace util
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
// Copyright (c) 2007-2012 Hartmut Kaiser
//
// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

//  See http://www.boost.org for updates, documentation, and revision history.

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_integral.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/string.hpp>
#include <boost/archive/archive_exception.hpp>
#include <boost/archive/detail/common_oarchive.hpp>
#include <boost/archive/detail/register_archive.hpp>
#if BOOST_VERSION >= 104400
#include <boost/serialization/item_version_type.hpp>
#endif

#include <hpx/config.hpp>
#include <hpx/util/portable_binary_archive.hpp>
#include <hpx/util/basic_binary_oprimitive.hpp>

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
// be passed across systems. Note:floating point types are passed through as is.

#if defined(BOOST_MSVC) || defined(BOOST_INTEL_WIN)
#define HPX_SERIALIZATION_EXPORT
#else
#define HPX_SERIALIZATION_EXPORT HPX_ALWAYS_EXPORT
#endif

class HPX_SERIALIZATION_EXPORT portable_binary_oarchive :
    public hpx::util::basic_binary_oprimitive<
        portable_binary_oarchive
    >,
    public boost::archive::detail::common_oarchive<
        portable_binary_oarchive
    >
{
    typedef hpx::util::basic_binary_oprimitive<
        portable_binary_oarchive
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
    HPX_ALWAYS_EXPORT void
    save_impl(const boost::intmax_t l, const char maxsize);
    // add base class to the places considered when matching
    // save function to a specific set of arguments.  Note, this didn't
    // work on my MSVC 7.0 system so we use the sure-fire method below
    // using archive_base_t::save;

    // default fall through for any types not specified here
    template <typename T>
    void save(T const& val, typename boost::enable_if<boost::is_integral<T> >::type* = 0) 
    {
        boost::intmax_t t = static_cast<boost::intmax_t>(val);
        save_impl(t, sizeof(T));
    }

    template <typename T>
    void save(T const& t, typename boost::disable_if<boost::is_integral<T> >::type* = 0) 
    {
        this->primitive_base_t::save(t);
    }

    void save(const std::string& t) {
        this->primitive_base_t::save(t);
    }
#if BOOST_VERSION >= 104400
    void save(const boost::archive::class_id_reference_type& t) {
        boost::intmax_t l = t;
        save_impl(l, sizeof(boost::int16_t));
    }
    void save(const boost::archive::class_id_optional_type& t) {
        boost::intmax_t l = t;
        save_impl(l, sizeof(boost::int16_t));
    }
    void save(const boost::archive::class_id_type& t) {
        boost::intmax_t l = t;
        save_impl(l, sizeof(boost::int16_t));
    }
    void save(const boost::archive::object_id_type& t) {
        boost::intmax_t l = t;
        save_impl(l, sizeof(boost::uint32_t));
    }
    void save(const boost::archive::object_reference_type& t) {
        boost::intmax_t l = t;
        save_impl(l, sizeof(boost::uint32_t));
    }
    void save(const boost::archive::tracking_type& t) {
        bool l = t;
        this->primitive_base_t::save(l);
    }
    void save(const boost::archive::version_type& t) {
        boost::intmax_t l = t;
        save_impl(l, sizeof(boost::uint32_t));
    }
    void save(const boost::archive::library_version_type& t) {
        boost::intmax_t l = t;
        save_impl(l, sizeof(boost::uint16_t));
    }
    void save(const boost::serialization::item_version_type& t) {
        boost::intmax_t l = t;
        save_impl(l, sizeof(boost::intmax_t));
    }
#endif
#ifndef BOOST_NO_STD_WSTRING
    void save(std::wstring const& t) {
        this->primitive_base_t::save(t);
    }
#endif
    void save(float const& t) {
        this->primitive_base_t::save(t);
    }
    void save(double const& t) {
        this->primitive_base_t::save(t);
    }
    void save(char const& t) {
        this->primitive_base_t::save(t);
    }
    void save(unsigned char const& t) {
        this->primitive_base_t::save(t);
    }
    void save(signed char const& t) {
        this->primitive_base_t::save(t);
    }


    // default processing - kick back to base class.  Note the
    // extra stuff to get it passed borland compilers
    typedef boost::archive::detail::common_oarchive<portable_binary_oarchive>
        detail_common_oarchive;

    template <typename T>
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

    HPX_ALWAYS_EXPORT void init(unsigned int flags);

public:
    portable_binary_oarchive(std::vector<char>& buffer, unsigned flags = 0)
      : primitive_base_t(buffer, flags & boost::archive::no_codecvt),
        archive_base_t(flags),
        m_flags(flags & (endian_big | endian_little))
    {
        init(flags);
    }

    // the optimized save_array dispatches to save_binary 

    // default fall through for any types not specified here
    template <typename T>
    void save_array(boost::serialization::array<T> const& a, unsigned int)
    {
        // If we need to potentially flip bytes we serialize each element 
        // separately.
#ifdef BOOST_BIG_ENDIAN
        if (m_flags & endian_little) {
            for (std::size_t i = 0; i != a.count(); ++i)
                save(a.address()[i]);
        }
#else
        if (m_flags & endian_big) {
            for (std::size_t i = 0; i != a.count(); ++i)
                save(a.address()[i]);
        }
#endif
        else {
            this->primitive_base_t::save_binary(a.address(), a.count()*sizeof(T));
        }
    }

    void save_array(boost::serialization::array<float> const& a, unsigned int)
    {
        this->primitive_base_t::save_binary(a.address(), a.count()*sizeof(float));
    }
    void save_array(boost::serialization::array<double> const& a, unsigned int)
    {
        this->primitive_base_t::save_binary(a.address(), a.count()*sizeof(double));
    }
    void save_array(boost::serialization::array<char> const& a, unsigned int)
    {
        this->primitive_base_t::save_binary(a.address(), a.count()*sizeof(char));
    }
    void save_array(boost::serialization::array<unsigned char> const& a, unsigned int)
    {
        this->primitive_base_t::save_binary(a.address(), a.count()*sizeof(unsigned char));
    }
    void save_array(boost::serialization::array<signed char> const& a, unsigned int)
    {
        this->primitive_base_t::save_binary(a.address(), a.count()*sizeof(signed char));
    }
};

#undef HPX_SERIALIZATION_EXPORT
}}

// required by export in boost version > 1.34
BOOST_SERIALIZATION_REGISTER_ARCHIVE(hpx::util::portable_binary_oarchive)
BOOST_SERIALIZATION_USE_ARRAY_OPTIMIZATION(hpx::util::portable_binary_oarchive)

#if defined(_MSC_VER)
#pragma warning( pop )
#endif

#endif // HPX_USE_PORTABLE_ARCHIVES == 0
#endif // PORTABLE_BINARY_OARCHIVE_HPP


