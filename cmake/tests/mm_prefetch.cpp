//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if defined(_MSC_VER)
#include <intrin.h>
#endif
#if defined(__GNUC__)
#include <emmintrin.h>
#endif

int main()
{
    char* p = 0;
    _mm_prefetch(p, _MM_HINT_T0);
}
