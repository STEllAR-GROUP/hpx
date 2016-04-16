////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_B72D9BF0_B236_46F6_83AA_E45A70BD1FAA)
#define HPX_B72D9BF0_B236_46F6_83AA_E45A70BD1FAA

#include <hpx/config.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/function.hpp>

#include <algorithm>
#include <deque>
#include <iterator>
#include <ostream>
#include <type_traits>
#include <vector>

namespace hpx { namespace iostreams
{

typedef util::function_nonser<void(std::vector<char> const&)> write_function_type;

///////////////////////////////////////////////////////////////////////////////
// Write function that works on STL OutputIterators
template <typename Iterator>
inline void iterator_write_function(std::vector<char> const& in, Iterator it)
{
    std::copy(in.begin(), in.end(), it);
}

// Factory function
template <typename Iterator>
inline write_function_type make_iterator_write_function(Iterator it)
{
    return util::bind(iterator_write_function<Iterator>,
        util::placeholders::_1, it);
}

///////////////////////////////////////////////////////////////////////////////
inline void
std_ostream_write_function(std::vector<char> const& in, std::ostream& os)
{
    std::copy(in.begin(), in.end(), std::ostream_iterator<char>(os));
}

// Factory function
inline write_function_type make_std_ostream_write_function(std::ostream& os)
{
    return util::bind(std_ostream_write_function,
        util::placeholders::_1, std::ref(os));
}

}}

#endif // HPX_B72D9BF0_B236_46F6_83AA_E45A70BD1FAA

