//  Copyright (c) 2014 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/string_util.hpp>

#include <iostream>
#include <iterator>
#include <regex>
#include <string>
#include <utility>
#include <vector>

struct pipeline
{
    static void process(std::vector<std::string> const& input)
    {
        // job for first stage
        auto grep = [](std::string const& re, std::string const& item) {
            std::regex regex(re);
            if (std::regex_match(item, regex))
            {
                auto trim = [](std::string const& s) {
                    return hpx::string_util::trim_copy(s);
                };

                hpx::async(trim, std::move(item))
                    .then(hpx::unwrapping([](std::string const& tc) {
                        std::cout << "->" << tc << std::endl;
                    }));
            }
        };

        std::vector<hpx::future<void>> tasks;
        for (auto s : input)
        {
            tasks.push_back(hpx::async(grep, "Error.*", std::move(s)));
        }

        hpx::wait_all(tasks);
    }
};

int hpx_main()
{
    std::string inputs[] = {"Error: foobar", "Error. foo", " Warning: barbaz",
        "Notice: qux", "\tError: abc"};
    std::vector<std::string> input(std::begin(inputs), std::end(inputs));

    pipeline::process(input);

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    return hpx::local::init(hpx_main, argc, argv);
}
