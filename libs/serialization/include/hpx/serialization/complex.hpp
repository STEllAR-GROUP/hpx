//  Copyright (c) 2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_COMPLEX_HPP
#define HPX_SERIALIZATION_COMPLEX_HPP

#include <hpx/serialization/serialization_fwd.hpp>
#include <hpx/serialization/traits/is_bitwise_serializable.hpp>

#include <complex>
#include <type_traits>

namespace hpx { namespace serialization {

    template <typename T>
    void serialize(input_archive& ar, std::complex<T>& c, unsigned)
    {
        T real, imag;
        ar >> real >> imag;
        c.real(real);
        c.imag(imag);
    }

    template <typename T>
    void serialize(output_archive& ar, std::complex<T> const& c, unsigned)
    {
        ar << c.real() << c.imag();
    }
}}    // namespace hpx::serialization

namespace hpx { namespace traits {

    template <typename T>
    struct is_bitwise_serializable<std::complex<T>>
      : is_bitwise_serializable<typename std::remove_const<T>::type>
    {
    };
}}    // namespace hpx::traits

#endif
