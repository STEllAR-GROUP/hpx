//  Unit test for hpx::util::lexical_cast.
//
//  See http://www.boost.org for most recent version, including documentation.
//
//  Copyright Sergey Shandar 2005, Alexander Nasonov, 2007.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost
//  Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt).
//
// Test abstract class. Bug 1358600:
// http://sf.net/tracker/?func=detail&aid=1358600&group_id=7586&atid=107586

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
#include <ostream>

using namespace hpx::util;

class A
{
public:
    virtual void out(std::ostream&) const = 0;
    virtual ~A() {}
};

class B : public A
{
public:
    virtual void out(std::ostream& O) const
    {
        O << "B";
    }
};

std::ostream& operator<<(std::ostream& O, const A& a)
{
    a.out(O);
    return O;
}

void test_abstract()
{
    const A& a = B();
    HPX_TEST(hpx::util::lexical_cast<std::string>(a) == "B");
}

int main()
{
    test_abstract();

    return hpx::util::report_errors();
}
