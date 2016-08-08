//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <boost/algorithm/string/trim.hpp>
#include <boost/range/functions.hpp>
#include <boost/regex.hpp>

struct pipeline
{
    static void process(std::vector<std::string> const& input)
    {
        // job for first stage
        auto grep = [](std::string const& re, std::string const& item)
        {
            boost::regex regex(re);
            if (boost::regex_match(item, regex))
            {
                auto trim = [](std::string const& s)
                {
                    return boost::algorithm::trim_copy(s);
                };

                hpx::async(trim, std::move(item))
                    .then(hpx::util::unwrapped(
                        [](std::string const& tc)
                        {
                            std::cout << "->" << tc << std::endl;
                        }));
            }
        };

        std::vector<hpx::future<void> > tasks;
        for(auto s: input)
        {
            tasks.push_back(hpx::async(grep, "Error.*", std::move(s)));
        }

        wait_all(tasks);
    }
};

int main()
{
    std::string inputs[] = {
        "Error: foobar",
        "Error. foo",
        " Warning: barbaz",
        "Notice: qux",
        "\tError: abc"
      };
    std::vector<std::string> input(boost::begin(inputs), boost::end(inputs));

    pipeline::process(input);

    return 0;
}

