////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <utility>

int main()
{
    std::integer_sequence<std::size_t, 0, 1, 2>* is = 0;
    std::index_sequence<0, 1, 2>* ss = 0;
    std::make_integer_sequence<std::size_t, 3>* mis = 0;
    std::make_index_sequence<3>* mss = 0;
    std::index_sequence_for<int, int, int>* misf = 0;

    is = ss = mis = mss = misf;
}
