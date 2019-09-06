//  Copyright (c) 2019 The STE||AR GROUP
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/program_options.hpp>
#include <hpx/program_options/force_linking.hpp>

namespace hpx { namespace program_options {
    force_linking_helper& force_linking()
    {
        static force_linking_helper helper
        {
#if !defined(HPX_PROGRAM_OPTIONS_HAVE_BOOST_PROGRAM_OPTIONS_COMPATIBILITY)
            &parse_environment, &parse_environment, &parse_environment,
                &parse_config_file<char>, &parse_config_file<char>,
                &parse_config_file<wchar_t>, &split_unix,
#endif
        };
        return helper;
    }
}}    // namespace hpx::program_options
