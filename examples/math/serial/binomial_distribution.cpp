////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <iostream>

#include <boost/cstdint.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>

double factorial(boost::uint64_t n)
{
    if (n <= 0)
        return 1;
    else
        return n * factorial(n - 1);
}

double binomial_distribution(boost::uint64_t n, double p, boost::uint64_t r)
{
    double fn  = factorial(n)
         , fnr = factorial(n - r)
         , fr  = factorial(r);

    return (fn / (fnr * fr)) * std::pow(p, r) * std::pow(1 - p, n - r);
} 

int main(int argc, char** argv)
{
    try
    {
        if (argc != 4)
            throw std::exception();

        const boost::uint64_t n = boost::lexical_cast<boost::uint64_t>(argv[1]);
        const double p = boost::lexical_cast<double>(argv[2]);
        const boost::uint64_t r = boost::lexical_cast<boost::uint64_t>(argv[3]);

        std::cout
            << ( boost::format("binomial_distribution(%1%, %2%, %3%) == %4%\n")
               % n % p % r % binomial_distribution(n, p, r));
    }

    catch (std::exception&)
    {
        std::cerr << (boost::format("Usage: %1% N P R\n") % argv[0]);
        return 1;
    }  
}

