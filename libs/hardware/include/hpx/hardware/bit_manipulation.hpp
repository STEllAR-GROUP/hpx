////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (C) 2010 Scott McMurray
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_68441787_8A39_4DAC_9D9E_705B6FA19BCD)
#define HPX_68441787_8A39_4DAC_9D9E_705B6FA19BCD

#include <cstddef>

namespace hpx { namespace util { namespace hardware
{

template <typename T, typename U>
bool has_bit_set(T value, U bit)
{ return (value & (1 << bit)) != 0; }

template <std::size_t N, typename T>
struct unbounded_shifter
{
    static T shl(T x) { return unbounded_shifter<N-1, T>::shl(T(x << 1)); }
    static T shr(T x) { return unbounded_shifter<N-1, T>::shr(T(x >> 1)); }
};

template <typename T>
struct unbounded_shifter<0, T>
{
    static T shl(T x) { return x; }
    static T shr(T x) { return x; }
};

template <std::size_t N, typename T>
T unbounded_shl(T x)
{ return unbounded_shifter<N, T>::shl(x); }

template <std::size_t N, typename T>
T unbounded_shr(T x)
{ return unbounded_shifter<N, T>::shr(x); }

template <std::size_t Low, std::size_t High, typename Result, typename T>
Result get_bit_range(T x)
{
    T highmask = unbounded_shl<High, T>(~T());
    T lowmask = unbounded_shl<Low, T>(~T());
    return static_cast<Result>
        (unbounded_shr<Low, T>(T(x & (lowmask ^ highmask))));
}

template <std::size_t Low, typename Result, typename T>
Result pack_bits(T x)
{ return unbounded_shl<Low, Result>(static_cast<Result>(x)); }

}}}

#endif // HPX_68441787_8A39_4DAC_9D9E_705B6FA19BCD

