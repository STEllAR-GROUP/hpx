////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>

#include <boost/array.hpp>

struct accumulator
{
  private:
    unsigned& total_;

  public:
    accumulator(
        unsigned& total
        )
      : total_(total)
    {}

    void operator()(
        unsigned x
        )
    {
        total_ += x;
    }
};

int main()
{
    boost::array<unsigned, 10> fib = { 0, 1, 1, 2, 3, 5, 8, 13, 21, 34 };

    unsigned total = 0;

    auto f = accumulator(total);

    std::for_each(fib.begin(), fib.end(), f);

    return !(88 == total);
}

