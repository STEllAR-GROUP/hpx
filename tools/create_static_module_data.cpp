//  Copyright (c) 2005-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config/defines.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <set>
#include <vector>
#include <algorithm>

#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>

#define REPLACE_TAG     "@HPX_REPLACE@"
#define NAME_TAG        "@COMPONENT_NAME@"
#define MODULE_TAG      "@MODULE_NAME@"

///////////////////////////////////////////////////////////////////////////////
bool split_module_line(std::string const& line, std::string& module,
    std::set<std::string>& components)
{
    using namespace boost::algorithm;
    std::vector<std::string> entries;
    split(entries, line, is_any_of(","));

    if (entries.empty())
        return false;

    module = entries[0];
    if (entries.size() > 1 && !entries[1].empty())
    {
        std::copy(++entries.begin(), entries.end(),
            std::inserter(components, components.end()));
    }
    return true;
}

///////////////////////////////////////////////////////////////////////////////
bool read_component_list(char const* listname, std::set<std::string>& components,
    std::set<std::string>& modules)
{
    std::ifstream module_list(listname);
    if (!module_list.is_open())
    {
        std::cerr << "create_static_module_data: could not not open: "
            << listname << "\n";
        return false;
    }

    std::string line;
    while (std::getline(module_list, line))
    {
        boost::algorithm::trim(line);
        if (line.empty() || line[0] == '#')
            continue;   // empty or comment line

        // split the line into module name and components
        std::string module;
        if (!split_module_line(line, module, components))
        {
            std::cerr << "create_static_module_data: split failed for line: "
                << line << "\n";
            return false;
        }

        modules.insert(module);
    }
    return true;
}

///////////////////////////////////////////////////////////////////////////////
bool replace_tags(std::ofstream& out, std::string templ,
    std::set<std::string> const& components,
    std::set<std::string> const& modules)
{
    {
        std::set<std::string>::const_iterator end = components.end();
        for (std::set<std::string>::const_iterator it = components.begin();
             it != end; ++it)
        {
            std::string::size_type p = templ.find(NAME_TAG);
            if (p != std::string::npos)
            {
                std::string s(templ);
                s.replace(p, sizeof(NAME_TAG)-1, *it);
                out << s;
            }
        }
    }

    {
        std::set<std::string>::const_iterator end = modules.end();
        for (std::set<std::string>::const_iterator it = modules.begin();
             it != end; ++it)
        {
            std::string::size_type p = templ.find(MODULE_TAG);
            if (p != std::string::npos)
            {
                std::string s(templ);
                s.replace(p, sizeof(MODULE_TAG)-1, *it);
                out << s;
            }
        }
    }
    return true;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // verify command line
    if (argc < 4)
    {
        std::cerr
            << "usage: create_static_module_data "
               "<infile> <outfile> <module_list1> [... <module_listN>]"
            << std::endl;
        return -1;
    }

    // open input file
    std::string inname(argv[1]);
    std::ifstream in(inname.c_str());
    if (!in.is_open())
    {
        std::cerr << "create_static_module_data: couldn't open input file: "
            << inname << std::endl;
        return -3;
    }

    // open output file
    std::string outname(argv[2]);
    std::ofstream out(outname.c_str());
    if (!out.is_open())
    {
        std::cerr << "create_static_module_data: couldn't open output file: "
            << outname << std::endl;
        return -4;
    }

    // read module list
    std::set<std::string> components, modules;
    for(int i = 3; i < argc; ++i)
    {
        if (!read_component_list(argv[i], components, modules))
        {
            std::cerr << "create_static_module_data: couldn't read module list: "
                << argv[i] << std::endl;
            return -2;
        }
    }

    // loop over the input lines and replace the tags
    int linenum = 0;
    std::string line;
    while (std::getline(in, line))
    {
        ++linenum;

        std::string::size_type p = line.find(REPLACE_TAG);
        if (p == std::string::npos)
        {
            out << line << std::endl;
            continue;
        }

        // output the rest of the line
        out << line.substr(0, p);
        std::string templ;

        std::string::size_type first = line.find("[[", p);
        if (first == std::string::npos)
        {
            std::cerr << "create_static_module_data: tag: " << REPLACE_TAG
                << " should be followed by '[['." << std::endl;
            return -5;
        }

        std::string::size_type last = line.find("]]", first);
        if (last != std::string::npos)
        {
            templ = line.substr(first+2, last-first-2);
        }
        else
        {
            templ = line.substr(first+2);
            templ += "\n";

            // find the line containing the closing parenthesis
            while (std::getline(in, line))
            {
                ++linenum;

                last = line.find("]]");
                if (last == std::string::npos)
                {
                    templ += line;
                    templ += "\n";
                }
                else
                {
                    templ += line.substr(0, last);
                    break;
                }
            }
        }

        if (templ[0] == '\n')
            templ.erase(0, 1);

        if (!replace_tags(out, templ, components, modules))
        {
            std::cerr << "create_static_module_data: failed to replace name "
                "tag at line: " << linenum << " (" << templ << ")" << std::endl;
        }
    }

    return 0;
}
