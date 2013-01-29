//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/function.hpp>
#include <boost/function.hpp>
#include <functional>

#include <hpx/util/high_resolution_timer.hpp>

unsigned iterations = 0;

struct foo
{
    void operator()() const
    {
        ++iterations;
    }
};

template <typename F>
void run(F const & f, unsigned max_iterations)
{
    hpx::util::high_resolution_timer t;
    iterations = 0;
    for(unsigned i = 0; i < max_iterations; ++i)
    {
        f();
    }
    std::cout << ", iterations per second: " << iterations/t.elapsed() << "\n";
}

int main()
{
    unsigned max_iterations = (std::numeric_limits<unsigned>::max)();
    {
        hpx::util::function<void(), void, void> f = foo();
        std::cout << "hpx::util::function";
        run(f, max_iterations);
    }
    {
        boost::function<void()> f = foo();
        std::cout << "boost::function";
        run(f, max_iterations);
    }
    {
        std::function<void()> f = foo();
        std::cout << "std::function";
        run(f, max_iterations);
    }
}
