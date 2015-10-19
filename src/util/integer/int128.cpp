//   Copyright: Copyright (C) 2005, Jan Ringos, http://Tringi.Mx-3.cz
//   Version: 1.1
//
//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/util/integer/int128.hpp>
#include <hpx/runtime/serialization/serialize.hpp>

#include <cmath>
#include <cstring>
#include <memory>
#include <algorithm>

// IMPLEMENTATION

namespace hpx { namespace util { namespace integer
{
    const char * int128::toString (unsigned int radix) const throw () {
        if (!*this) return "0";
        if (radix < 2 || radix > 37) return "(invalid radix)";

        static char sz [256];
        std::memset (sz, 0, 256);

        int128 r;
        int128 ii = (*this < 0) ? -*this : *this;
        int i = 255;

        while (!!ii && i) {
            ii = ii.div (radix, r);
            sz [--i] = r.toInt () + ((r.toInt () > 9) ? 'A' - 10 : '0');
        };

        if (*this < 0)
            sz [--i] = '-';

        return &sz [i];
    };

    int128::int128 (const char * sz) throw ()
        : lo (0u), hi (0) {

        if (!sz) return;
        if (!sz [0]) return;

        unsigned int radix = 10;
        std::size_t i = 0;
        bool minus = false;

        if (sz [i] == '-') {
            ++i;
            minus = true;
        };

        if (sz [i] == '0') {
            radix = 8;
            ++i;
            if (sz [i] == 'x') {
                radix = 16;
                ++i;
            };
        };

        std::size_t len = strlen (sz);
        for (; i < len; ++i) {
            unsigned int n = 0;
            if (sz [i] >= '0' && sz [i] <= (std::min)(('0' + (int)radix), (int)'9'))
                n = sz [i] - '0';
            else if (sz [i] >= 'a' && sz [i] <= 'a' + (int) radix - 10)
                n = sz [i] - 'a' + 10;
            else if (sz [i] >= 'A' && sz [i] <= 'A' + (int) radix - 10)
                n = sz [i] - 'A' + 10;
            else
                break;

            (*this) *= radix;
            (*this) += n;
        };

        if (minus)
            *this = 0 - *this;

        return;
    };

    int128::int128 (const float a) throw ()
        : lo ((boost::uint64_t) fmodf (a, 18446744073709551616.0f)),
          hi ((boost::int64_t) (a / 18446744073709551616.0f)) {};

    int128::int128 (const double & a) throw ()
        : lo ((boost::uint64_t) fmod (a, 18446744073709551616.0)),
          hi ((boost::int64_t) (a / 18446744073709551616.0)) {};

    int128::int128 (const long double & a) throw ()
        : lo ((boost::uint64_t) fmodl (a, 18446744073709551616.0l)),
          hi ((boost::int64_t) (a / 18446744073709551616.0l)) {};

    float int128::toFloat () const throw () {
        return (float) this->hi * 18446744073709551616.0f
             + (float) this->lo;
    };

    double int128::toDouble () const throw () {
        return (double) this->hi * 18446744073709551616.0
             + (double) this->lo;
    };

    long double int128::toLongDouble () const throw () {
        return (long double) this->hi * 18446744073709551616.0l
             + (long double) this->lo;
    };

    int128 int128::operator - () const throw () {
        if (!this->hi && !this->lo)
            // number is 0, just return 0
            return *this;

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4146)
#endif
        // non 0 number
        return int128 (-this->lo, ~this->hi);
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
    };

    int128 int128::operator ~ () const throw () {
        return int128 (~this->lo, ~this->hi);
    };

    int128 & int128::operator ++ () {
        ++this->lo;
        if (!this->lo)
            ++this->hi;

        return *this;
    };

    int128 & int128::operator -- () {
        if (!this->lo)
            --this->hi;
        --this->lo;

        return *this;
    };

    int128 int128::operator ++ (int) {
        int128 b = *this;
        ++ *this;

        return b;
    };

    int128 int128::operator -- (int) {
        int128 b = *this;
        -- *this;

        return b;
    };

    int128 & int128::operator += (const int128 & b) throw () {
        boost::uint64_t old_lo = this->lo;

        this->lo += b.lo;
        this->hi += b.hi + (this->lo < old_lo);

        return *this;
    };

    int128 & int128::operator *= (const int128 & b) throw () {
        if (!b)
            return *this = 0u;
        if (b == 1u)
            return *this;

        int128 a = *this;
        int128 t = b;

        this->lo = 0ull;
        this->hi = 0ll;

        for (unsigned int i = 0; i < 128; ++i) {
            if (t.lo & 1)
                *this += a << i;

            t >>= 1;
        };

        return *this;
    };


    int128 int128::div (const int128 & divisor, int128 & remainder) const throw () {
        if (!divisor)
            return 1u / (unsigned int) divisor.lo;
            // or RaiseException (EXCEPTION_INT_DIVIDE_BY_ZERO,
            //                    EXCEPTION_NONCONTINUABLE, 0, NULL);

        int128 ds = (divisor < 0) ? -divisor : divisor;
        int128 dd = (*this < 0) ? -*this : *this;

        // only remainder
        if (ds > dd) {
            remainder = *this;
            return boost::int64_t(0ll);
        };

        int128 r = boost::int64_t(0ll);
        int128 q = boost::int64_t(0ll);
    //    while (dd >= ds) { dd -= ds; q += 1; }; // extreme slow version

        unsigned int b = 127;
        while (r < ds) {
            r <<= 1;
            if (dd.bit (b--))
                r.lo |= 1;
        };
        ++b;

        while (true)
            if (r < ds) {
                if (!(b--)) break;

                r <<= 1;
                if (dd.bit (b))
                    r.lo |= 1;

            } else {
                r -= ds;
                q.bit (b, true);
            };

        // correct
        if ((divisor < 0) ^ (*this < 0)) q =- q;
        if (*this < 0) r =- r;

        remainder = r;
        return q;
    };

    bool int128::bit (unsigned int n) const throw () {
        n &= 0x7F;

        if (n < 64)
            return (this->lo & (1ull << n)) ? true : false;
        return (this->hi & (1ull << (n - 64))) ? true : false;
    };

    void int128::bit (unsigned int n, bool val) throw () {
        n &= 0x7F;

        if (val) {
            if (n < 64) this->lo |= (1ull << n);
                   else this->hi |= (1ull << (n - 64));
        } else {
            if (n < 64) this->lo &= ~(1ull << n);
                   else this->hi &= ~(1ull << (n - 64));
        };
    };


    int128 & int128::operator >>= (unsigned int n) throw () {
        n &= 0x7F;

        if (n > 63) {
            n -= 64;
            this->lo = this->hi;

            if (this->hi < 0) this->hi = -1ll;
                         else this->hi = 0ll;
        };

        if (n) {
            // shift low qword
            this->lo >>= n;

            // get lower N bits of high qword
            boost::uint64_t mask = 0ull;
            for (unsigned int i = 0; i < n; ++i)
                mask |= (1ll << i);

            // and add them to low qword
            this->lo |= (this->hi & mask) << (64 - n);

            // and finally shift also high qword
            this->hi >>= n;
        };

        return *this;
    };

    int128 & int128::operator <<= (unsigned int n) throw () {
        n &= 0x7F;

        if (n > 63) {
            n -= 64;
            this->hi = this->lo;
            this->lo = 0ull;
        };

        if (n) {
            // shift high qword
            this->hi <<= n;

            // get higher N bits of low qword
            boost::uint64_t mask = 0ull;
            for (unsigned int i = 0; i < n; ++i)
                mask |= (1ll << (63 - i));

            // and add them to high qword
            this->hi |= (this->lo & mask) >> (64 - n);

            // and finally shift also low qword
            this->lo <<= n;
        };

        return *this;
    };

    bool int128::operator ! () const throw () {
        return !(this->hi || this->lo);
    };

    int128 & int128::operator |= (const int128 & b) throw () {
        this->hi |= b.hi;
        this->lo |= b.lo;

        return *this;
    };

    int128 & int128::operator &= (const int128 & b) throw () {
        this->hi &= b.hi;
        this->lo &= b.lo;

        return *this;
    };

    int128 & int128::operator ^= (const int128 & b) throw () {
        this->hi ^= b.hi;
        this->lo ^= b.lo;

        return *this;
    };

    bool operator <  (const int128 & a, const int128 & b) throw () {
        if (a.hi == b.hi) {
            if (a.hi < 0)
                return (boost::int64_t) a.lo < (boost::int64_t) b.lo;
            else
                return a.lo < b.lo;
        } else
            return a.hi < b.hi;
    };

    bool operator == (const int128 & a, const int128 & b) throw () {
        return a.hi == b.hi && a.lo == b.lo;
    };
    bool operator && (const int128 & a, const int128 & b) throw () {
        return (a.hi || a.lo) && (b.hi || b.lo);
    };
    bool operator || (const int128 & a, const int128 & b) throw () {
        return (a.hi || a.lo) || (b.hi || b.lo);
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Archive>
    void int128::save(Archive& ar, const unsigned int version) const
    {
        boost::int64_t hi_ = hi;
        boost::uint64_t lo_ = lo;
        ar & lo_ & hi_;
    }

    template void int128::save<serialization::output_archive>(
        serialization::output_archive& ar, const unsigned int version) const;

    template <typename Archive>
    void int128::load(Archive& ar, const unsigned int version)
    {
        boost::int64_t hi_ = 0;
        boost::uint64_t lo_ = 0;
        ar & lo_ & hi_;
        hi = hi_;
        lo = lo_;
    }

    template void int128::load<serialization::input_archive>(
        serialization::input_archive& ar, const unsigned int version);
}}}

