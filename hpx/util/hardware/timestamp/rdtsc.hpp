//  Copyright (c) 2011 Vinay C Amatya
//
//Parts of the code have been adapted from Katzumo's (ANL) version on rdtsc



#ifndef RDTSC_HPP_
#define RDTSC_HPP_

#include <stdint.h>

#if defined (_WIN32)
    #include <Windows.h>
    #include <intrin.h>
    #define U_LARGE_INT unsigned __int64

    U_LARGE_INT inline get_rdtsc()
    {
        LARGE_INTEGER ticks;
        QueryPerformanceCounter(&ticks);
        return (U_LARGE_INT)(ticks.QuadPart);
    }
#elif defined(__unix__)
    #define U_LARGE_INT uint64_t

    #if defined(__i386__)
        static __inline__ U_LARGE_INT get_rdtsc()
        {
            U_LARGE_INT x;
            __asm pushad; //save all registers
            __asm mov eax, 0;
            cpuid;
            __asm__ volatile__ (".byte 0x0f, 0x32" : "=A" (x));
            return x;
        }

    #elif defined(__x86_64__)
        static __inline__ U_LARGE_INT get_rdtsc()
        {
            uint32_t hi, lo;
            __asm__ __volatile__ (
                "xorl %%eax,%%eax \n        cpuid"
                ::: "%rax", "%rbx", "%rcx", "%rdx");
        // We cannot use "=A", since this would use %rax on x86_64 
        //and return only the lower 	32bits of the TSC */
        __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
        return (uint64_t)hi << 32 | lo;

        }

    #endif

#endif

#endif
