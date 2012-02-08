
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  Ported from OpenMPI C code ... 

#ifndef HPX_EXAMPLES_BRIGHT_FUTURE_CREATE_DIM_HPP
#define HPX_EXAMPLES_BRIGHT_FUTURE_CREATE_DIM_HPP

#include <boost/array.hpp>
#include <vector>
#include <algorithm>
#include <functional>

inline std::vector<std::size_t> assign_locals(std::size_t dim, std::vector<std::size_t> const & primes, std::vector<std::size_t> const & factors)
{
    typedef std::vector<std::size_t> vector_type;
    typedef vector_type::iterator iterator;

    vector_type locals(dim, 1);
    // Loop assigning factors from the highest to the lowest
    for(std::size_t j = 0; j < primes.size(); ++j)
    {
        std::size_t f = primes[j];
        for(std::size_t n = factors[j]; n > 0; --n)
        {
            iterator min_bin = std::min_element(locals.begin(), locals.end());
            *min_bin *= f;
        }
    }

    std::sort(locals.begin(), locals.end(), std::greater<std::size_t>());

    return locals;
}

inline std::vector<std::size_t> get_factors(std::size_t nlocals, std::vector<std::size_t> & primes)
{
    if(primes.size() == 0)
    {
        throw "...";
    }

    typedef std::vector<std::size_t> vector_type;
    typedef vector_type::reverse_iterator reverse_iterator;

    vector_type factors(primes.size());

    for(reverse_iterator c = factors.rbegin(), p = primes.rbegin(); c != factors.rend(); ++c, ++p)
    {
        *c = 0;
        while((nlocals % *p) == 0)
        {
            ++(*c);
            nlocals /= *p;
        }
    }

    if(nlocals != 1 )
    {
        throw "...";
    }

    return factors;
}

inline std::vector<std::size_t> get_primes(std::size_t nlocals)
{
    typedef std::vector<std::size_t> vector_type;
    typedef vector_type::reverse_iterator reverse_iterator;

    std::size_t size = (nlocals/2)+1;
    vector_type primes;
    primes.reserve(size);

    std::size_t i = 0;
    ++i;
    primes.push_back(2);

    for(std::size_t n = 3; n <= nlocals; n += 2)
    {
        std::size_t j = 1;
        for(; j < i; ++j)
        {
            if((n % primes[j]) == 0)
            {
                break;
            }
        }
        if(j == i)
        {
            if(i >= size)
            {
                throw "...";
            }
            ++i;
            primes.push_back(n);
        }
    }

    return primes;
}

inline std::vector<std::size_t> create_dim(std::size_t nlocals, std::size_t dim)
{
    if(dim == 1)
    {
        return std::vector<std::size_t>(1, nlocals);
    }

    std::vector<std::size_t> primes = get_primes(nlocals);
    std::vector<std::size_t> factors = get_factors(nlocals, primes);
    return assign_locals(dim, primes, factors);
}

#endif
