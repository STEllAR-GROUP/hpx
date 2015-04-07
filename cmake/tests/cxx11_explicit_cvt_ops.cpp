////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2008 Beman Dawes
//  Copyright (c) 2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

struct foo {
    explicit operator int() const { return 1; }
};

int main()
{
    foo f;
    int i = int(f);
}
