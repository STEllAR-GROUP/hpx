////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2013 Andrey Semashev
//  Copyright (c) 2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

// 190022816: VS2015RC
#if defined(_MSC_FULL_VER) && _MSC_FULL_VER <= 190022816
#error "Inline namespaces are broken on VS2015 (versions below don't support it)"
#endif

inline namespace my_ns {

    int data = 0;
}

int main()
{
    data = 1;
    my_ns::data = 1;
}
