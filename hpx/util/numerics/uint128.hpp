//   Copyright: Copyright (C) 2005, Jan Ringos, http://Tringi.Mx-3.cz
//   Version: 1.1
//
//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef UINT128_HPP
#define UINT128_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_UINT128)
#include <hpx/hpx_fwd.hpp>

#include <exception>
#include <cstdlib>
#include <cstdio>
#include <new>

namespace hpx { namespace util { namespace numerics
{
    class HPX_EXPORT uint128
    {
    private:
        // Binary correct representation of unsigned 128bit integer
        unsigned __int64    lo;
        unsigned __int64    hi;

    protected:
        // Some global operator functions must be friends
        friend bool operator <  (const uint128 &, const uint128 &) throw ();
        friend bool operator == (const uint128 &, const uint128 &) throw ();
        friend bool operator || (const uint128 &, const uint128 &) throw ();
        friend bool operator && (const uint128 &, const uint128 &) throw ();

        #ifdef __GNUC__
            friend uint128 operator <? (const uint128 &, const uint128 &) throw ();
            friend uint128 operator >? (const uint128 &, const uint128 &) throw ();
        #endif

    public:
        // Constructors
        inline uint128 () throw () {};
        inline uint128 (const uint128 & a) throw () : lo (a.lo), hi (a.hi) {};

        inline uint128 (const unsigned int & a) throw () : lo (a), hi (0ull) {};
        inline uint128 (const unsigned __int64 & a) throw () : lo (a), hi (0ull) {};

        uint128 (const float a) throw ();
        uint128 (const double & a) throw ();
        uint128 (const long double & a) throw ();

        uint128 (const char * sz) throw ();

        // TODO: Consider creation of operator= to eliminate
        //       the need of intermediate objects during assignments.

    private:
        // Special internal constructors
        uint128 (const unsigned __int64 & a, const unsigned __int64 & b) throw ()
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
        __int64 toUint64 () const throw () {
            return (unsigned __int64) this->lo; };
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
        friend class boost::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int version);
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

#ifdef __GNUC__
        inline uint128 operator <? (const uint128 & a, const uint128 & b) throw () {
            return (a < b) ? a : b; };
        inline uint128 operator >? (const uint128 & a, const uint128 & b) throw () {
            return (a < b) ? b : a; };
#endif

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

typedef hpx::util::numerics::uint128 __uint128_t;

#endif
#endif
