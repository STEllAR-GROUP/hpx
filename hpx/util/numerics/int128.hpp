//   Copyright: Copyright (C) 2005, Jan Ringos, http://Tringi.Mx-3.cz
//   Version: 1.1
//
//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef INT128_HPP
#define INT128_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_UINT128)
#include <hpx/hpx_fwd.hpp>

#include <exception>
#include <cstdlib>
#include <cstdio>
#include <new>

// CLASS
namespace hpx { namespace util { namespace numerics
{
    class HPX_EXPORT int128
    {
    private:
        // Binary correct representation of signed 128bit integer
        unsigned __int64    lo;
        signed   __int64    hi;

    protected:
        // Some global operator functions must be friends
        friend bool operator <  (const int128 &, const int128 &) throw ();
        friend bool operator == (const int128 &, const int128 &) throw ();
        friend bool operator || (const int128 &, const int128 &) throw ();
        friend bool operator && (const int128 &, const int128 &) throw ();

        #ifdef __GNUC__
            friend int128 operator <? (const int128 &, const int128 &) throw ();
            friend int128 operator >? (const int128 &, const int128 &) throw ();
        #endif

    public:
        // Constructors
        inline int128 () throw () {};
        inline int128 (const int128 & a) throw () : lo (a.lo), hi (a.hi) {};

        inline int128 (const unsigned int & a) throw () : lo (a), hi (0ll) {};
        inline int128 (const signed int & a) throw () : lo (a), hi (0ll) {
            if (a < 0) this->hi = -1ll;
        };

        inline int128 (const unsigned __int64 & a) throw () : lo (a), hi (0ll) {};
        inline int128 (const signed __int64 & a) throw () : lo (a), hi (0ll) {
            if (a < 0) this->hi = -1ll;
        };

        int128 (const float a) throw ();
        int128 (const double & a) throw ();
        int128 (const long double & a) throw ();

        int128 (const char * sz) throw ();

        // TODO: Consider creation of operator= to eliminate
        //       the need of intermediate objects during assignments.

    private:
        // Special internal constructors
        int128 (const unsigned __int64 & a, const signed __int64 & b) throw ()
            : lo (a), hi (b) {};

    public:
        // Operators
        bool operator ! () const throw ();

        int128 operator - () const throw ();
        int128 operator ~ () const throw ();

        int128 & operator ++ ();
        int128 & operator -- ();
        int128 operator ++ (int);
        int128 operator -- (int);

        int128 & operator += (const int128 & b) throw ();
        int128 & operator *= (const int128 & b) throw ();

        int128 & operator >>= (unsigned int n) throw ();
        int128 & operator <<= (unsigned int n) throw ();

        int128 & operator |= (const int128 & b) throw ();
        int128 & operator &= (const int128 & b) throw ();
        int128 & operator ^= (const int128 & b) throw ();

        // Inline simple operators
        inline const int128 & operator + () const throw () { return *this; };

        // Rest of inline operators
        inline int128 & operator -= (const int128 & b) throw () {
            return *this += (-b);
        };
        inline int128 & operator /= (const int128 & b) throw () {
            int128 dummy;
            *this = this->div (b, dummy);
            return *this;
        };
        inline int128 & operator %= (const int128 & b) throw () {
            this->div (b, *this);
            return *this;
        };

        // Common methods
        int toInt () const throw () {  return (int) this->lo; };
        __int64 toInt64 () const throw () {  return (__int64) this->lo; };

        const char * toString (unsigned int radix = 10) const throw ();
        float toFloat () const throw ();
        double toDouble () const throw ();
        long double toLongDouble () const throw ();

        // Arithmetic methods
        int128  div (const int128 &, int128 &) const throw ();

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

    bool operator <  (const int128 & a, const int128 & b) throw ();
    bool operator == (const int128 & a, const int128 & b) throw ();
    bool operator || (const int128 & a, const int128 & b) throw ();
    bool operator && (const int128 & a, const int128 & b) throw ();

#ifdef __GNUC__
        inline int128 operator <? (const int128 & a, const int128 & b) throw () {
            return (a < b) ? a : b; };
        inline int128 operator >? (const int128 & a, const int128 & b) throw () {
            return (a < b) ? b : a; };
#endif

    // GLOBAL OPERATOR INLINES

    inline int128 operator + (const int128 & a, const int128 & b) throw () {
        return int128 (a) += b; };
    inline int128 operator - (const int128 & a, const int128 & b) throw () {
        return int128 (a) -= b; };
    inline int128 operator * (const int128 & a, const int128 & b) throw () {
        return int128 (a) *= b; };
    inline int128 operator / (const int128 & a, const int128 & b) throw () {
        return int128 (a) /= b; };
    inline int128 operator % (const int128 & a, const int128 & b) throw () {
        return int128 (a) %= b; };

    inline int128 operator >> (const int128 & a, unsigned int n) throw () {
        return int128 (a) >>= n; };
    inline int128 operator << (const int128 & a, unsigned int n) throw () {
        return int128 (a) <<= n; };

    inline int128 operator & (const int128 & a, const int128 & b) throw () {
        return int128 (a) &= b; };
    inline int128 operator | (const int128 & a, const int128 & b) throw () {
        return int128 (a) |= b; };
    inline int128 operator ^ (const int128 & a, const int128 & b) throw () {
        return int128 (a) ^= b; };

    inline bool operator >  (const int128 & a, const int128 & b) throw () {
        return   b < a; };
    inline bool operator <= (const int128 & a, const int128 & b) throw () {
        return !(b < a); };
    inline bool operator >= (const int128 & a, const int128 & b) throw () {
        return !(a < b); };
    inline bool operator != (const int128 & a, const int128 & b) throw () {
        return !(a == b); };
}}}

// MISC

typedef hpx::util::numerics::int128 __int128_t;

#endif
#endif
