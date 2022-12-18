//  Copyright (c) 2019 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

struct alignas(64) overaligned
{
};

int main()
{
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(error: 4316)
#elif defined(__GNUC__)
#pragma GCC diagnostic push
// NOTE: The actual error we're looking for here is triggered by -Waligned-new,
// but GCC does not allow us to turn only that into an error.
#pragma GCC diagnostic error "-Wall"
#endif

    overaligned* s = new overaligned;

#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

    delete s;
    return 0;
}
