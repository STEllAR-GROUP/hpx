#ifndef PORTABLE_BINARY_IARCHIVE_HPP
#define PORTABLE_BINARY_IARCHIVE_HPP

// MS compatible compilers support #pragma once
#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

/////////1/////////2/////////3/////////4/////////5/////////6/////////7/////////8
// portable_binary_iarchive.hpp

// (C) Copyright 2002 Robert Ramey - http://www.rrsd.com . 
// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

//  See http://www.boost.org for updates, documentation, and revision history.

#include <ostream>
#include <algorithm>
#include <climits>
#include <boost/version.hpp>
#include <boost/archive/binary_iarchive.hpp>
#if BOOST_VERSION >= 103500
#include <boost/archive/shared_ptr_helper.hpp>
#else
#include <boost/serialization/shared_ptr.hpp>
#endif
#include <boost/detail/endian.hpp>
#include <boost/cstdint.hpp>

namespace hpx { namespace util
{

/////////1/////////2/////////3/////////4/////////5/////////6/////////7/////////8
// exception to be thrown if integer read from archive doesn't fit
// variable being loaded
class portable_binary_archive_exception : 
    public virtual boost::archive::archive_exception
{
public:
    typedef enum {
        incompatible_integer_size 
    } exception_code;
    portable_binary_archive_exception(exception_code c = incompatible_integer_size)
    {}
    virtual const char *what( ) const throw( )
    {
        const char *msg = "programmer error";
        switch(code){
        case incompatible_integer_size:
            msg = "integer cannot be represented";
        default:
            boost::archive::archive_exception::what();
        }
        return msg;
    }
};

/////////1/////////2/////////3/////////4/////////5/////////6/////////7/////////8
// "Portable" input binary archive.  This is a variation of the native binary 
// archive. it addresses integer size and endianness so that binary archives can
// be passed across systems. Note:floating point types not addressed here
class portable_binary_iarchive :
    // don't derive from binary_iarchive !!!
    public boost::archive::binary_iarchive_impl<
        portable_binary_iarchive, 
        std::istream::char_type, 
        std::istream::traits_type
    >
#if BOOST_VERSION >= 103500
  , public boost::archive::detail::shared_ptr_helper
#else
  , public boost::serialization::detail::shared_ptr_helper
#endif
{
    typedef boost::archive::binary_iarchive_impl<
        portable_binary_iarchive, 
        std::istream::char_type, 
        std::istream::traits_type
    > archive_base_t;
    typedef boost::archive::basic_binary_iprimitive<
        portable_binary_iarchive, 
        std::ostream::char_type, 
        std::ostream::traits_type
    > primitive_base_t;
#ifndef BOOST_NO_MEMBER_TEMPLATE_FRIENDS
public:
#else
    friend archive_base_t;
    friend primitive_base_t; // since with override load below
    friend class boost::archive::basic_binary_iarchive<portable_binary_iarchive>;
    friend class boost::archive::load_access;
#endif
    void load_impl(boost::intmax_t& l, char maxsize){
        boost::uint8_t size;
        this->archive_base_t::load(size);
        if(size > maxsize || size > sizeof(boost::intmax_t))
            throw portable_binary_archive_exception();
        l = 0;
        load_binary(&l, size);

// we choose to use big endian (hey it's the network byte ordering)
#ifdef BOOST_LITTLE_ENDIAN
        boost::int8_t* first = 
            static_cast<boost::int8_t*>(static_cast<void*>(&l));
        boost::int8_t* last = first + size - 1;
        for(/**/ ; first < last; ++first, --last)
            std::swap(*first, *last);
#endif

        // extend sign if necessary 
        if (size != maxsize && (l >> (size - 1) * CHAR_BIT) & 0x80) 
            l |= (-1 << (size * CHAR_BIT));
    }
    // default fall through for any types not specified here
    template<class T>
    void load(T & t){
        this->primitive_base_t::load(t);
    }
    void load(unsigned short & t){
        boost::intmax_t l;
        load_impl(l, sizeof(unsigned short));
        t = l;
    }
    void load(short & t){
        boost::intmax_t l;
        load_impl(l, sizeof(short));
        t = l;
    }
    void load(unsigned int & t){
        boost::intmax_t l;
        load_impl(l, sizeof(unsigned int));
        t = l;
    }
    void load(int & t){
        boost::intmax_t l;
        load_impl(l, sizeof(int));
        t = l;
    }
    void load(unsigned long & t){
        boost::intmax_t l;
        load_impl(l, sizeof(unsigned long));
        t = l;
    }
    void load(long & t){
        boost::intmax_t l;
        load_impl(l, sizeof(long));
        t = l;
    }
#if defined(BOOST_HAS_LONG_LONG)
    void load(boost::long_long_type & t){
        boost::intmax_t l;
        load_impl(l, sizeof(boost::long_long_type));
        t = l;
    }
    void load(boost::ulong_long_type & t){
        boost::intmax_t l;
        load_impl(l, sizeof(boost::ulong_long_type));
        t = l;
    }
#endif
    void load(float& t){
        BOOST_ASSERT(false);    // floating point types are not implemented
    }
    void load(double& t){
        BOOST_ASSERT(false);    // floating point types are not implemented
    }
    void load(long double& t){
        BOOST_ASSERT(false);    // floating point types are not implemented
    }
public:
    portable_binary_iarchive(std::istream & is, unsigned flags = 0) :
        archive_base_t(
            is, 
            flags | boost::archive::no_header // skip default header checking 
        )
    {
        // use our own header checking
        if(0 != (flags & boost::archive::no_header)){
            this->archive_base_t::init(flags);
            // skip the following for "portable" binary archives
            // boost::archive::basic_binary_oprimitive<derived_t, std::ostream>::init();
        }
    }
};

}}  // namespace hpx::util

#ifdef BOOST_SERIALIZATION_REGISTER_ARCHIVE
    BOOST_SERIALIZATION_REGISTER_ARCHIVE(hpx::util::portable_binary_iarchive)
#endif

// required by export in boost <= 1.34
#define BOOST_ARCHIVE_CUSTOM_IARCHIVE_TYPES hpx::util::portable_binary_iarchive

// explicitly instantiate for this type of text stream
#include <boost/archive/impl/basic_binary_iarchive.ipp>
#include <boost/archive/impl/archive_pointer_iserializer.ipp>
#include <boost/archive/impl/basic_binary_iprimitive.ipp>

namespace boost { namespace archive 
{
    template class binary_iarchive_impl<
        hpx::util::portable_binary_iarchive, 
        std::istream::char_type, 
        std::istream::traits_type
    >;
    template class detail::archive_pointer_iserializer<
        hpx::util::portable_binary_iarchive>;

}} // namespace boost::archive

#endif // PORTABLE_BINARY_IARCHIVE_HPP
