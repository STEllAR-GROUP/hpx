//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/testing/performance.hpp>

#include <chrono>
#include <cstddef>
#include <iostream>
#include <string>
#include <type_traits>

namespace hpx { namespace util {

    namespace detail {

        json_perf_times& times()
        {
            static json_perf_times res;
            return res;
        }

        void add_time(std::string const& test_name, std::string const& executor,
            double time)
        {
            times().add(test_name, executor, time);
        }

        HPX_CORE_EXPORT std::ostream& operator<<(
            std::ostream& strm, json_perf_times const& obj)
        {
            strm << "{\n";
            strm << "  \"outputs\" : [";
            int outputs = 0;
            for (auto&& item : obj.m_map)
            {
                if (outputs)
                    strm << ",";
                strm << "\n    {\n";
                strm << "      \"name\" : \"" << std::get<0>(item.first)
                     << "\",\n";
                strm << "      \"executor\" : \"" << std::get<1>(item.first)
                     << "\",\n";
                strm << "      \"series\" : [";
                int series = 0;
                for (auto val : item.second)
                {
                    if (series)
                        strm << ", ";
                    strm << val;
                    ++series;
                }
                strm << "]\n";
                strm << "    }";
                ++outputs;
            }
            if (outputs)
                strm << "\n  ";
            strm << "]\n";
            strm << "}\n";
            return strm;
        }
    }    // namespace detail

    void perftests_report(std::string const& name, std::string const& exec,
        const std::size_t steps, hpx::function<void(void)>&& test)
    {
        if (steps == 0)
            return;
        // First iteration to cache the data
        test();
        using timer = std::chrono::high_resolution_clock;
        timer::time_point start;
        for (size_t i = 0; i != steps; ++i)
        {
            // For now we don't flush the cache
            //flush_cache();
            start = timer::now();
            test();
            // default is in seconds
            auto time =
                std::chrono::duration_cast<std::chrono::duration<double>>(
                    timer::now() - start);
            detail::add_time(name, exec, time.count());
        }
    }

    void perftests_print_times()
    {
        std::cout << detail::times();
    }
}}    // namespace hpx::util
