
#include <hpx/hpx_init.hpp>
#include <boost/assign/std.hpp>

#include <iostream>
#include <string>

#include "./str_srch/server/str_search.hpp"
#include "./str_srch/text_split.hpp"

using boost::program_options::variables_map;
using boost::program_options::options_description;

////////////////////////////////////////////////////////////////////////////////

int hpx_main(boost::program_options::variables_map& vm)
{
    {
        char char_replace = 'a';
        std::string result_string, input_string;

        input_string = "The best weapon of a dictatorship is secrecy, but the best weapon of a democracy should be the weapon of openness.";        
        std::cout << input_string << std::endl;
        text::text_split ts;
        ts.create();
        result_string = ts.process(char_replace, input_string);

        std::cout << result_string << std::endl;
    }
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using boost::program_options::value;

    //Application specific options
    options_description cmdline("Usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX
    return hpx::init(cmdline, argc, argv);
}

