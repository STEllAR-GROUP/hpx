////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Kohei Takahashi
//  Copyright (c) 2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

constexpr void decrement(int& value) { --value; }

struct S
{
    constexpr S() : ptr(nullptr) {}
    constexpr void reset() { ptr = nullptr; }
    constexpr void reset() const { }
    void * ptr;
};

constexpr int zero()
{
    int ret = 1;
    decrement(ret);

    S s;
    s.reset();

    return ret;
}

int main()
{
    constexpr int i = zero();
    S s;
    s.reset();
}
