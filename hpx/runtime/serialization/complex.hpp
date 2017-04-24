//  Copyright (c) 2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_COMPLEX_HPP
#define HPX_SERIALIZATION_COMPLEX_HPP

#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/traits/is_bitwise_serializable.hpp>

#include <complex>
#include <type_traits>

namespace hpx { namespace serialization
{
    template <typename T>
    void serialize(input_archive& ar, std::complex<T>& c, unsigned)
    {
        T real, imag;
        ar >> real >> imag;
        c.real(real);
        c.imag(imag);
    }

    template <typename T>
    void serialize(output_archive& ar, const std::complex<T>& c, unsigned)
    {
        ar << c.real() << c.imag();
    }
}}

namespace hpx { namespace traits
{
    template <typename T>
    struct is_bitwise_serializable<std::complex<T> >
      : is_bitwise_serializable<typename std::remove_const<T>::type>
    {};
}}

#endif
