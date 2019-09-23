//  Copyright (c) 2019 The STE||AR GROUP
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config/force_linking.hpp>
#include <hpx/config/version.hpp>

namespace hpx { namespace config {
    // reference all symbols that have to be explicitly linked with the core
    // library
    force_linking_helper& force_linking()
    {
        static force_linking_helper helper{
            hpx::HPX_CHECK_VERSION, hpx::HPX_CHECK_BOOST_VERSION};
        return helper;
    }
}}    // namespace hpx::config
