////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_B72D9BF0_B236_46F6_83AA_E45A70BD1FAA)
#define HPX_B72D9BF0_B236_46F6_83AA_E45A70BD1FAA

#include <ostream>
#include <iterator>
#include <algorithm>
#include <deque>

#include <boost/ref.hpp>
#include <boost/bind.hpp>
#include <boost/bind/placeholders.hpp>

namespace hpx { namespace iostreams
{

typedef HPX_STD_FUNCTION<void(std::deque<char> const&)> write_function_type;

///////////////////////////////////////////////////////////////////////////////
// Write function that works on STL OutputIterators
template <typename Iterator>
inline void iterator_write_function(std::deque<char> const& in, Iterator it)
{ std::copy(in.begin(), in.end(), it); }

// Factory function
template <typename Iterator>
inline write_function_type make_iterator_write_function(Iterator it)
{
    return write_function_type(boost::bind
        (iterator_write_function<Iterator>, _1, it));
}

///////////////////////////////////////////////////////////////////////////////
inline void
std_ostream_write_function(std::deque<char> const& in, std::ostream& os)
{ std::copy(in.begin(), in.end(), std::ostream_iterator<char>(os)); }

// Factory function
inline write_function_type make_std_ostream_write_function(std::ostream& os)
{
    return write_function_type(boost::bind
        (std_ostream_write_function, _1, boost::ref(os)));
}

}}

#endif // HPX_B72D9BF0_B236_46F6_83AA_E45A70BD1FAA

