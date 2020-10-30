//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/modules/testing.hpp>

#include <sstream>
#include <string>

int main()
{
    std::stringstream strm;

    // testing macros
    HPX_TEST(true);
    HPX_TEST(strm, true);

    HPX_TEST_MSG(true, "should be true");
    HPX_TEST_MSG(strm, true, "should be true");

    HPX_TEST_EQ(0, 0);
    HPX_TEST_EQ(strm, 0, 0);

    HPX_TEST_EQ_MSG(0, 0, "should be equal");
    HPX_TEST_EQ_MSG(strm, 0, 0, "should be equal");

    HPX_TEST_NEQ(0, 1);
    HPX_TEST_NEQ(strm, 0, 1);

    HPX_TEST_NEQ_MSG(0, 1, "should not be equal");
    HPX_TEST_NEQ_MSG(strm, 0, 1, "should not be equal");

    HPX_TEST_LT(0, 1);
    HPX_TEST_LT(strm, 0, 1);

    HPX_TEST_LT_MSG(0, 1, "should be less");
    HPX_TEST_LT_MSG(strm, 0, 1, "should be less");

    HPX_TEST_LTE(1, 1);
    HPX_TEST_LTE(strm, 1, 1);

    HPX_TEST_LTE_MSG(1, 1, "should be less equal");
    HPX_TEST_LTE_MSG(strm, 1, 1, "should be less equal");

    HPX_TEST_RANGE(1, 1, 1);
    HPX_TEST_RANGE(strm, 1, 1, 1);

    HPX_TEST_RANGE_MSG(1, 1, 1, "should be in range");
    HPX_TEST_RANGE_MSG(strm, 1, 1, 1, "should be in range");

    // sanity macro tests
    HPX_SANITY(true);
    HPX_SANITY(strm, true);

    HPX_SANITY_MSG(true, "should be true");
    HPX_SANITY_MSG(strm, true, "should be true");

    HPX_SANITY_EQ(0, 0);
    HPX_SANITY_EQ(strm, 0, 0);

    HPX_SANITY_EQ_MSG(0, 0, "should be equal");
    HPX_SANITY_EQ_MSG(strm, 0, 0, "should be equal");

    HPX_SANITY_NEQ(0, 1);
    HPX_SANITY_NEQ(strm, 0, 1);

    HPX_SANITY_LT(0, 1);
    HPX_SANITY_LT(strm, 0, 1);

    HPX_SANITY_LTE(1, 1);
    HPX_SANITY_LTE(strm, 1, 1);

    HPX_SANITY_RANGE(1, 1, 1);
    HPX_SANITY_RANGE(strm, 1, 1, 1);

    // there shouldn't be any output being generated
    HPX_TEST(strm.str().empty());

    // now test that something gets written to the stream if an error occurs
    HPX_TEST(strm, false);
    HPX_TEST(strm.str().find("test 'false'") != std::string::npos);

    // we have intentionally generated one error
    return (hpx::util::report_errors() == 1) ? 0 : -1;
}
