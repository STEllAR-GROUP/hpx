////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Takaya Saito
//  Copyright (c) 2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

int f() noexcept;
int g() noexcept(noexcept(f()));

int main()
{
    bool b = noexcept(g());
}
