//  Copyright (c) 2019 National Technology & Engineering Solutions of Sandia,
//                     LLC (NTESS).
//  Copyright (c) 2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/modules/resiliency.hpp>

#include <iostream>

int main(int argc, char* argv[])
{
    std::cout << "HPX Resiliency module version: "
              << hpx::resiliency::experimental::full_version_str() << "\n";
    return 0;
}
