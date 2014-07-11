//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>

#include <iostream>
#include <string>

#include <boost/algorithm/string/trim.hpp>

///////////////////////////////////////////////////////////////////////////////
hpx::future<std::string> read()
{
    std::string s("   foo   ");
    return hpx::make_ready_future(s);
}

hpx::future<std::string> process(std::string const& input)
{
    return
        hpx::async(
            [](std::string const& s)
                { return boost::algorithm::trim_copy(s); },
            input
        );
}

hpx::future<void> write(std::string const& input)
{
    return
        hpx::async(
            [](std::string const& s)
                { std::cout << "->" << s << "<-" << std::endl; },
            input
        );
}

///////////////////////////////////////////////////////////////////////////////
struct future_identity
{
    template <typename Future>
    Future operator() (Future && f) const
    {
        return f;
    }
};

template <typename F1, typename F2>
auto operator& (F1 && f1, F2 && f2) -> decltype(
        hpx::util::bind(future_identity(), f1().then(hpx::util::unwrapped(f2)))
    )
{
    return hpx::util::bind(future_identity(),
        f1().then(hpx::util::unwrapped(f2)));
}

int main()
{
    auto read_step = []() { return read(); };

    auto result = read_step & process & write;
    result();

    return 0;
}

