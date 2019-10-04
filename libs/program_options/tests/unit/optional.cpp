//  Copyright Vladimir Prus 2002-2004.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/testing.hpp>

#include <hpx/program_options.hpp>

#include <hpx/datastructures/optional.hpp>

#include <string>
#include <vector>

namespace po = hpx::program_options;

std::vector<std::string> sv(const char* array[], unsigned size)
{
    std::vector<std::string> r;
    for (unsigned i = 0; i < size; ++i)
        r.emplace_back(array[i]);
    return r;
}

void test_optional()
{
// Support for storing into optionals was added in 1.65.0
#if !defined(HPX_PROGRAM_OPTIONS_HAVE_BOOST_PROGRAM_OPTIONS_COMPATIBILITY) ||  \
    (defined(BOOST_VERSION) && BOOST_VERSION > 106500)
    po::optional<int> foo, bar, baz;

    po::options_description desc;
    // clang-format off
    desc.add_options()
        ("foo,f", po::value(&foo), "")
        ("bar,b", po::value(&bar), "")
        ("baz,z", po::value(&baz), "")
        ;
    // clang-format on

    const char* cmdline1_[] = {"--foo=12", "--bar", "1"};
    std::vector<std::string> cmdline1 =
        sv(cmdline1_, sizeof(cmdline1_) / sizeof(const char*));

    po::variables_map vm;
    po::store(po::command_line_parser(cmdline1).options(desc).run(), vm);
    po::notify(vm);

    HPX_TEST(!!foo);
    HPX_TEST(*foo == 12);

    HPX_TEST(!!bar);
    HPX_TEST(*bar == 1);

    HPX_TEST(!baz);
#endif
}

int main(int, char*[])
{
    test_optional();
    return hpx::util::report_errors();
}
