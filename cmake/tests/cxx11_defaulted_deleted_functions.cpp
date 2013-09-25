////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <cstdint>
#include <utility>

struct trivial {
    trivial() = default;
    trivial(const trivial&) = default;
    trivial(trivial&&) = default;
    trivial& operator=(const trivial&) = default;
    trivial& operator=(trivial&&) = default;
    ~trivial() = default;
};
struct onlydouble {
    onlydouble() = delete; // OK, but redundant
    onlydouble(std::intmax_t) = delete;
    onlydouble(double);
};

struct moveonly {
    moveonly() = default;
    moveonly(const moveonly&) = delete;
    moveonly(moveonly&&) = default;
    moveonly& operator=(const moveonly&) = delete;
    moveonly& operator=(moveonly&&) = default;
    ~moveonly() = default;
};

int main()
{
    moveonly p;
    moveonly q(p); // error, deleted copy constructor
    moveonly r(std::move(p)); // ok, defaulted move constructor
}
