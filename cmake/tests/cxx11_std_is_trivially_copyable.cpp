////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <type_traits>

int main()
{
    int check_is_triviallycopyable[std::is_trivially_copyable<int>::value ? 1 : -1];
}
