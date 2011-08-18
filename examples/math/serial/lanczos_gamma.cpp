////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <complex>

#include <boost/cstdint.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>
#include <boost/array.hpp>
#include <boost/math/constants/constants.hpp>

typedef std::complex<double> complex_type;

struct lanczos_gamma_f
{
    template <typename T>
    static complex_type call(T z);

    static const boost::array<double, 9> p; 
    static const double pi;
};

const boost::array<double, 9> lanczos_gamma_f::p = { {
    0.99999999999980993
  , 676.5203681218851
  , -1259.1392167224028
  , 771.32342877765313
  , -176.61502916214059
  , 12.507343278686905
  , -0.13857109526572012
  , 9.9843695780195716e-6
  , 1.5056327351493116e-7
} };

const double lanczos_gamma_f::pi = boost::math::constants::pi<double>();

template <typename T>
complex_type lanczos_gamma_f::call(T z)
{
    complex_type zc(z);

    if (std::real(zc) < 0.5)
        return pi / (std::sin(pi * zc) * call(complex_type(1) - zc));

    else
    {
        complex_type x = p[0];

        zc -= 1;

        for (boost::array<double, 9>::size_type i = 1; i < p.size(); ++i)
            x += p[i] / (zc + complex_type(i));

        complex_type t = zc + complex_type(p.size() - 2) + 0.5;

        return std::sqrt(2 * pi) * std::pow(t, (zc + 0.5)) * std::exp(-t) * x;
    }
}

// forwarder
template <typename T>
complex_type lanczos_gamma(T z)
{ return lanczos_gamma_f::call(z); }

int main(int argc, char** argv)
{
    try
    {
        if (2 != argc)
            throw std::exception();

        const double z = boost::lexical_cast<double>(argv[1]);

        complex_type r = lanczos_gamma(z);

        std::cout
            << ( boost::format("lanczos_gamma(%1%) == %2%+%3%j\n")
               % z % std::real(r) % std::imag(r));
    }

    catch (std::exception&)
    {
        std::cerr << (boost::format("Usage: %1% Z\n") % argv[0]);
        return 1;
    }  
}

