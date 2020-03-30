////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/bind_back.hpp>
#include <hpx/functional/function.hpp>

#include <algorithm>
#include <deque>
#include <functional>
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
    return util::bind_back(iterator_write_function<Iterator>, it);
}

///////////////////////////////////////////////////////////////////////////////
inline void
std_ostream_write_function(std::vector<char> const& in, std::ostream& os)
{
    std::copy(in.begin(), in.end(), std::ostream_iterator<char>(os));
    os.flush();
}

// Factory function
inline write_function_type make_std_ostream_write_function(std::ostream& os)
{
    return util::bind_back(std_ostream_write_function, std::ref(os));
}

}}


