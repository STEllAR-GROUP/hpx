//  Copyright (c) 2020-2022 STE||AR Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/type_support/unused.hpp>

#include <cstddef>
#include <string>

#if (defined(WIN32) || defined(_WIN32) || defined(__WIN32__)) &&               \
    !defined(HPX_MINGW)

void detail_set_thread_name(const char*)
{
    // set_thread_name is guarenteed to exist on windows
    return;
}
#elif (defined(__linux__))

#include <pthread.h>

void detail_set_thread_name(const char* thread_name)
{
    pthread_setname_np(pthread_self(), thread_name);
    return;
}
#elif (defined(__NetBSD__))

#include <pthread.h>

void detail_set_thread_name(const char* thread_name)
{
    pthread_setname_np(pthread_self(), thread_name, NULL);
    return;
}

#elif (defined(__FreeBSD__) || defined(__OpenBSD__))

#include <pthread.h>

void detail_set_thread_name(const char* thread_name)
{
    pthread_set_name_np(pthread_self(), thread_name);
    return;
}

#endif

int main()
{
    detail_set_thread_name("hpx_thread_1");
}
