////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_A3492DD6_D911_4B12_94AB_7CDA3A03B223)
#define HPX_A3492DD6_D911_4B12_94AB_7CDA3A03B223

#include <boost/rational.hpp>

#include <boost/serialization/split_free.hpp>

namespace boost { namespace serialization
{

template <typename Archive, typename T>
void save(Archive& ar, boost::rational<T> const& r, const unsigned int)
{
    T num = r.numerator(), den = r.denominator(); 
    ar & num;
    ar & den;
}

template <typename Archive, typename T>
void load(Archive& ar, boost::rational<T>& r, const unsigned int)
{
    T num(0), den(0);
    ar & num;
    ar & den;
    r.assign(num, den);
}

template <typename Archive, typename T>
void serialize(Archive& ar, boost::rational<T>& r, const unsigned int version)
{ split_free(ar, r, version); }

}}

#endif // HPX_A3492DD6_D911_4B12_94AB_7CDA3A03B223

