//  Unit test for hpx::util::lexical_cast.
//
//  See http://www.boost.org for most recent version, including documentation.
//
//  Copyright Alexander Nasonov, 2007.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost
//  Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt).
//
//  Test that Source can be non-copyable.

#include <hpx/config.hpp>

#if defined(__INTEL_COMPILER)
#pragma warning(disable : 193 383 488 981 1418 1419)
#elif defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(                                                               \
    disable : 4097 4100 4121 4127 4146 4244 4245 4511 4512 4701 4800)
#endif

#include <hpx/lexical_cast.hpp>
#include <hpx/testing.hpp>

#include <string>

using namespace hpx::util;

class Noncopyable
{
    HPX_NON_COPYABLE(Noncopyable);

public:
    Noncopyable() {}
};

inline std::ostream& operator<<(std::ostream& out, const Noncopyable&)
{
    return out << "Noncopyable";
}

void test_noncopyable()
{
    Noncopyable x;
    HPX_TEST(hpx::util::lexical_cast<std::string>(x) == "Noncopyable");
}

int main()
{
    test_noncopyable();

    return hpx::util::report_errors();
}
