//   Copyright: Copyright (C) 2005, Jan Ringos, http://Tringi.Mx-3.cz
//   Version: 1.1
//
//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_INTEGER_UINT128_HPP
#define HPX_UTIL_INTEGER_UINT128_HPP

#include <hpx/config/export_definitions.hpp>
#include <hpx/runtime/serialization/serialization_fwd.hpp>

#include <exception>
#include <cstdlib>
#include <cstdio>
#include <new>

#include <boost/cstdint.hpp>

namespace hpx { namespace util { namespace integer
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_EXPORT uint128
    {
    private:
        // Binary correct representation of unsigned 128bit integer
        boost::uint64_t lo;
        boost::uint64_t hi;

    protected:
        // Some global operator functions must be friends
        friend bool operator <  (const uint128 &, const uint128 &) throw ();
        friend bool operator == (const uint128 &, const uint128 &) throw ();
        friend bool operator || (const uint128 &, const uint128 &) throw ();
        friend bool operator && (const uint128 &, const uint128 &) throw ();

    public:
        // Constructors
        inline uint128 () throw () {};

        inline uint128 (const unsigned int & a) throw () : lo (a), hi (0ull) {};
        inline uint128 (const boost::uint64_t & a) throw () : lo (a), hi (0ull) {};

        uint128 (const float a) throw ();
        uint128 (const double & a) throw ();
        uint128 (const long double & a) throw ();

        uint128 (const char * sz) throw ();

        // TODO: Consider creation of operator= to eliminate
        //       the need of intermediate objects during assignments.

    private:
        // Special internal constructors
        uint128 (const boost::uint64_t & a, const boost::uint64_t & b) throw ()
            : lo (a), hi (b) {};

    public:
        // Operators
        bool operator ! () const throw ();

        uint128 operator - () const throw ();
        uint128 operator ~ () const throw ();

        uint128 & operator ++ ();
        uint128 & operator -- ();
        uint128 operator ++ (int);
        uint128 operator -- (int);

        uint128 & operator += (const uint128 & b) throw ();
        uint128 & operator *= (const uint128 & b) throw ();

        uint128 & operator >>= (unsigned int n) throw ();
        uint128 & operator <<= (unsigned int n) throw ();

        uint128 & operator |= (const uint128 & b) throw ();
        uint128 & operator &= (const uint128 & b) throw ();
        uint128 & operator ^= (const uint128 & b) throw ();

        // Inline simple operators
        inline const uint128 & operator + () const throw () { return *this; };

        // Rest of inline operators
        inline uint128 & operator -= (const uint128 & b) throw () {
            return *this += (-b);
        };
        inline uint128 & operator /= (const uint128 & b) throw () {
            uint128 dummy;
            *this = this->div (b, dummy);
            return *this;
        };
        inline uint128 & operator %= (const uint128 & b) throw () {
            this->div (b, *this);
            return *this;
        };

        // Common methods
        unsigned int toUint () const throw () {
            return (unsigned int) this->lo; };
        boost::uint64_t toUint64 () const throw () {
            return (boost::uint64_t) this->lo; };
        const char * toString (unsigned int radix = 10) const throw ();
        float toFloat () const throw ();
        double toDouble () const throw ();
        long double toLongDouble () const throw ();

        // Arithmetic methods
        uint128  div (const uint128 &, uint128 &) const throw ();

        // Bit operations
        bool    bit (unsigned int n) const throw ();
        void    bit (unsigned int n, bool val) throw ();

    private:
        friend class hpx::serialization::access;

        HPX_SERIALIZATION_SPLIT_MEMBER()

        template <typename Archive>
        void save(Archive& ar, const unsigned int version) const;
        template <typename Archive>
        void load(Archive& ar, const unsigned int version);
    }
    #ifdef __GNUC__
        __attribute__ ((__aligned__ (16), __packed__))
    #endif
    ;


    // GLOBAL OPERATORS

    bool operator <  (const uint128 & a, const uint128 & b) throw ();
    bool operator == (const uint128 & a, const uint128 & b) throw ();
    bool operator || (const uint128 & a, const uint128 & b) throw ();
    bool operator && (const uint128 & a, const uint128 & b) throw ();

    // GLOBAL OPERATOR INLINES

    inline uint128 operator + (const uint128 & a, const uint128 & b) throw () {
        return uint128 (a) += b; };
    inline uint128 operator - (const uint128 & a, const uint128 & b) throw () {
        return uint128 (a) -= b; };
    inline uint128 operator * (const uint128 & a, const uint128 & b) throw () {
        return uint128 (a) *= b; };
    inline uint128 operator / (const uint128 & a, const uint128 & b) throw () {
        return uint128 (a) /= b; };
    inline uint128 operator % (const uint128 & a, const uint128 & b) throw () {
        return uint128 (a) %= b; };

    inline uint128 operator >> (const uint128 & a, unsigned int n) throw () {
        return uint128 (a) >>= n; };
    inline uint128 operator << (const uint128 & a, unsigned int n) throw () {
        return uint128 (a) <<= n; };

    inline uint128 operator & (const uint128 & a, const uint128 & b) throw () {
        return uint128 (a) &= b; };
    inline uint128 operator | (const uint128 & a, const uint128 & b) throw () {
        return uint128 (a) |= b; };
    inline uint128 operator ^ (const uint128 & a, const uint128 & b) throw () {
        return uint128 (a) ^= b; };

    inline bool operator >  (const uint128 & a, const uint128 & b) throw () {
        return   b < a; };
    inline bool operator <= (const uint128 & a, const uint128 & b) throw () {
        return !(b < a); };
    inline bool operator >= (const uint128 & a, const uint128 & b) throw () {
        return !(a < b); };
    inline bool operator != (const uint128 & a, const uint128 & b) throw () {
        return !(a == b); };
}}}

// MISC

// typedef hpx::util::integer::uint128 __uint128_t;

#endif
