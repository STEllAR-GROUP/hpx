#ifndef PORTABLE_BINARY_OARCHIVE_HPP
#define PORTABLE_BINARY_OARCHIVE_HPP

// MS compatible compilers support #pragma once
#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

/////////1/////////2/////////3/////////4/////////5/////////6/////////7/////////8
// portable_binary_oarchive.hpp

// (C) Copyright 2002 Robert Ramey - http://www.rrsd.com . 
// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

//  See http://www.boost.org for updates, documentation, and revision history.

#include <ostream>
#include <algorithm>
#include <climits>
#include <boost/archive/archive_exception.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/detail/endian.hpp>
#include <boost/cstdint.hpp>

namespace hpx { namespace util
{

/////////1/////////2/////////3/////////4/////////5/////////6/////////7/////////8
// "Portable" output binary archive.  This is a variation of the native binary 
// archive. it addresses integer size and endianness so that binary archives can
// be passed across systems. Note:floating point types not addressed here

class portable_binary_oarchive :
    // don't derive from binary_oarchive !!!
    public boost::archive::binary_oarchive_impl<
        portable_binary_oarchive, 
        std::ostream::char_type, 
        std::ostream::traits_type
    >
{
    typedef boost::archive::binary_oarchive_impl<
        portable_binary_oarchive, 
        std::ostream::char_type, 
        std::ostream::traits_type
    > archive_base_t;
    typedef boost::archive::basic_binary_oprimitive<
        portable_binary_oarchive, 
        std::ostream::char_type, 
        std::ostream::traits_type
    > primitive_base_t;
#ifndef BOOST_NO_MEMBER_TEMPLATE_FRIENDS
public:
#else
    friend archive_base_t;
    friend primitive_base_t; // since with override save below
    friend class boost::archive::basic_binary_oarchive<portable_binary_oarchive>;
    friend class boost::archive::save_access;
#endif
    void save_impl(boost::intmax_t l, char maxsize)
    {
        boost::intmax_t ll = l;
        boost::uint8_t size = 1;
        if (l < 0) {
            // make sure that enough of data is output
            // to include a high order bit indicating the sign
            do {
                ll >>= CHAR_BIT;
                ++size;
            } while (ll != -1);
        }
        else {
            do {
                ll >>= CHAR_BIT;
                ++size;
            } while (ll != 0);
        }
        if (size > maxsize)
            size = maxsize;

        this->archive_base_t::save(size);

// we choose to use little endian (it's more common)
#ifdef BOOST_BIG_ENDIAN
        boost::int8_t* first = 
            static_cast<boost::int8_t*>(static_cast<void*>(&l));
        boost::int8_t* last = first + size - 1;
        for(/**/; first < last; ++first, --last)
            std::swap(*first, *last);
#endif
        save_binary(&l, size);
    }
    
    template <typename T>
    void save_impl_fp(T l)
    {
        boost::uint8_t size = sizeof(T);
        this->archive_base_t::save(size);

// we choose to use little endian (it's more common)
#ifdef BOOST_BIG_ENDIAN
        boost::int8_t* first = 
            static_cast<boost::int8_t*>(static_cast<void*>(&l));
        boost::int8_t* last = first + size - 1;
        for(/**/; first < last; ++first, --last)
            std::swap(*first, *last);
#endif
        save_binary(&l, size);
    }

    // add base class to the places considered when matching
    // save function to a specific set of arguments.  Note, this didn't
    // work on my MSVC 7.0 system so we use the sure-fire method below
    // using archive_base_t::save;

    // default fall through for any types not specified here
    template<class T>
    void save(const T & t){
        this->primitive_base_t::save(t);
    }
    void save(const short t){
        save_impl(t, sizeof(short));
    }
    void save(const unsigned short t){
        save_impl(t, sizeof(unsigned short));
    }
    void save(const unsigned int t){
        save_impl(t, sizeof(unsigned int));
    }
    void save(const int t){
        save_impl(t, sizeof(int));
    }
    void save(const unsigned long t){
        save_impl(t, sizeof(unsigned long));
    }
    void save(const long t){
        save_impl(t, sizeof(long));
    }
#if defined(BOOST_HAS_LONG_LONG)
    void save(boost::long_long_type const t){
        save_impl(t, sizeof(boost::long_long_type));
    }
    void save(boost::ulong_long_type const t){
        save_impl(t, sizeof(boost::ulong_long_type));
    }
#endif
    void save(const float t){
        save_impl_fp(t);
    }
    void save(const double t){
        save_impl_fp(t);
    }
    void save(const long double t){
        save_impl_fp(t);
    }
public:
    portable_binary_oarchive(std::ostream & os, unsigned flags = 0) :
        archive_base_t(
            os, 
            flags | boost::archive::no_header // skip default header checking 
        )
    {
        // use our own header checking
        if(0 != (flags & boost::archive::no_header)){
            this->archive_base_t::init(flags);
            // skip the following for "portable" binary archives
            // boost::archive::basic_binary_iprimitive<derived_t, std::ostream>::init();
        }
    }
};

}}  // namespace hpx::util

#ifdef BOOST_SERIALIZATION_REGISTER_ARCHIVE
    BOOST_SERIALIZATION_REGISTER_ARCHIVE(hpx::util::portable_binary_oarchive)
#endif

#endif // PORTABLE_BINARY_OARCHIVE_HPP
