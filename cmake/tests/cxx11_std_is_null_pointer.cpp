////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <cstddef>
#include <type_traits>

int main()
{
    int check_nullptr[std::is_null_pointer<std::nullptr_t>::value ? 1 : -1];
}
