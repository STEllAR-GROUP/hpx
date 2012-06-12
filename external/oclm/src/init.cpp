
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <oclm/init.hpp>
#include <oclm/host.hpp>

#include <boost/program_options.hpp>

namespace oclm {
    bool init()
    {
        return init(std::vector<std::string>());
    }

    bool init(int argc, char ** argv)
    {
        return init(std::vector<std::string>(argv, argv + argc));
    }

    bool init(std::vector<std::string> const & args)
    {
        using boost::program_options::command_line_parser;
        using boost::program_options::options_description;
        using boost::program_options::variables_map;
        using boost::program_options::store;

        options_description desc("Allowed options");
        
        variables_map vm;        
        command_line_parser p(args);
        p.options(desc);

        store(p.run(), vm);
        notify(vm);


        host::get();
        
        return true;
    }
}
