////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <chrono>

int main()
{
    using namespace std::chrono;

    seconds s(1);
    nanoseconds ns = duration_cast<seconds>(s);
}
