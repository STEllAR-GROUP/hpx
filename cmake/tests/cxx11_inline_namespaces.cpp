////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2013 Andrey Semashev
//  Copyright (c) 2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

inline namespace my_ns {

    int data = 0;
}

int main()
{
    data = 1;
    my_ns::data = 1;
}
