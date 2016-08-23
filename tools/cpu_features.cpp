////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <cstddef>
#include <iomanip>
#include <ios>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include <hpx/util/hardware/cpuid.hpp>

using hpx::util::hardware::cpuid;
using hpx::util::hardware::cpu_info;
using hpx::util::hardware::cpuid_table;
using hpx::util::hardware::cpuid_table_type;
using hpx::util::hardware::has_bit_set;

using boost::program_options::variables_map;
using boost::program_options::positional_options_description;
using boost::program_options::options_description;
using boost::program_options::command_line_parser;
using boost::program_options::value;
using boost::program_options::notify;
using boost::program_options::store;

namespace {

struct return_value
{
    enum info
    {
        success                  = 0,
        feature_not_found        = 1,
        help                     = 2,
        list                     = 3,
        no_features_specified    = 4,
        unknown_feature          = 5,
        std_exception_thrown     = 6,
        unknown_exception_thrown = 7
    };
};

}

int main(int argc, char* argv[])
{
    try {
        options_description visible
            ("Usage: " HPX_APPLICATION_STRING " [options]");
        visible.add_options()
            ("help", "produce help message")
            ("quiet,q", "don't print results")
            ("list", "list known features")
            ;

        options_description hidden("Hidden options");
        hidden.add_options()
            ("features", value<std::vector<std::string> >(), "features to test")
            ;

        options_description cmdline_options;
        cmdline_options.add(visible).add(hidden);

        positional_options_description p;
        p.add("features", -1);

        variables_map vm;
        store(command_line_parser(argc, argv).
              options(cmdline_options).positional(p).run(), vm);
        notify(vm);

        if (vm.count("help"))
        {
            std::cout << visible << "\n";
            return return_value::help;
        }

        if (vm.count("list"))
        {
            std::cout << "known features:\n";
            for (std::size_t i = 0;
                 i < (sizeof(cpuid_table) / sizeof(cpuid_table_type));
                 ++i)
            { std::cout << "  " << cpuid_table[i].name << "\n"; }
            return return_value::list;
        }

        if (!vm.count("features"))
        {
            if (!vm.count("quiet"))
                std::cerr << "error: no features specified!\n\n"
                          << visible << "\n";
            return return_value::no_features_specified;
        }

        std::vector<std::string> raw_features
            = vm["features"].as<std::vector<std::string> >();

        std::set<std::string> features(raw_features.begin(),
                                       raw_features.end());

        cpu_info registers = { 0, 0, 0, 0 };

        bool not_found = false;

        for (std::size_t i = 0;
             i < (sizeof(cpuid_table) / sizeof(cpuid_table_type));
             ++i)
        {
            if (features.count(cpuid_table[i].name))
            {
                cpuid(registers, cpuid_table[i].function);
                bool found =  has_bit_set(registers[cpuid_table[i].register_],
                                          cpuid_table[i].bit);
                if (!vm.count("quiet"))
                    std::cout << std::setfill(' ') << std::setw(10) << std::left
                              << cpuid_table[i].name
                              << found << "\n";

                if (!found)
                    not_found = true;

                features.erase(cpuid_table[i].name);
            }
        }

        if (!features.empty())
        {
            if (!vm.count("quiet"))
            {
                for (std::string const& e : features)
                { std::cerr << "error: unknown feature '" << e << "'!\n"; }

                std::cerr << "\n" << visible << "\n";
            }

            return return_value::unknown_feature;
        }

        if (not_found)
            return return_value::feature_not_found;
    }

    catch (std::exception& e)
    {
        std::cout << "error: " << e.what() << "\n";
        return return_value::std_exception_thrown;
    }

    catch (...)
    {
        std::cout << "error: unknown exception occurred!\n";
        return return_value::unknown_exception_thrown;
    }

    return return_value::success;
}

