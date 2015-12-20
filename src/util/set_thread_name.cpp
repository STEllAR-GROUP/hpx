//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  This code was partially taken from:
//      http://msdn.microsoft.com/en-us/library/xcb2z8hs.aspx

#include <hpx/config.hpp>

#if defined(HPX_MSVC)

#include <hpx/util/set_thread_name.hpp>

namespace hpx { namespace util
{
    DWORD const MS_VC_EXCEPTION = 0x406D1388;

#pragma pack(push,8)
    typedef struct tagTHREADNAME_INFO
    {
       DWORD dwType;      // Must be 0x1000.
       LPCSTR szName;     // Pointer to name (in user addr space).
       DWORD dwThreadID;  // Thread ID (-1=caller thread).
       DWORD dwFlags;     // Reserved for future use, must be zero.
    } THREADNAME_INFO;
#pragma pack(pop)

    // Set the name of the thread shown in the Visual Studio debugger
    void set_thread_name(char const* threadName, DWORD dwThreadID)
    {
        THREADNAME_INFO info;
        info.dwType = 0x1000;
        info.szName = threadName;
        info.dwThreadID = dwThreadID;
        info.dwFlags = 0;

        __try
        {
            RaiseException(MS_VC_EXCEPTION, 0, sizeof(info)/sizeof(ULONG_PTR),
                (ULONG_PTR*)&info);
        }
        __except(EXCEPTION_EXECUTE_HANDLER)
        {
        }
    }
}}

#endif
