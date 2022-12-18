//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/testing/performance.hpp>

#include <chrono>
#include <cstddef>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace hpx::util {

    namespace detail {

        // Json output for performance reports
        class json_perf_times
        {
            using key_t = std::tuple<std::string, std::string>;
            using value_t = std::vector<double>;
            using map_t = std::map<key_t, value_t>;

            map_t m_map;

            HPX_CORE_EXPORT friend std::ostream& operator<<(
                std::ostream& strm, json_perf_times const& obj);

        public:
            HPX_CORE_EXPORT void add(std::string const& name,
                std::string const& executor, double time);
        };

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

        std::ostream& operator<<(std::ostream& strm, json_perf_times const& obj)
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

        void json_perf_times::add(
            std::string const& name, std::string const& executor, double time)
        {
            m_map[key_t(name, executor)].push_back(time);
        }
    }    // namespace detail

    void perftests_report(std::string const& name, std::string const& exec,
        std::size_t const steps, hpx::function<void(void)>&& test)
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
}    // namespace hpx::util
