////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2017 Marcin Copik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////


int main()
{
    int n = 0, m = 0;
    switch(n)
    {
        case 0:
            ++m;
        [[fallthrough]];
        case 1:
            m += 2;
        break;
    }
}
