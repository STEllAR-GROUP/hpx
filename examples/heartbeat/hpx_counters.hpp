//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/* This file was auto-generated from
C:\CVS\hpx\hpx\examples\heartbeat\heartbeat.man by ctrpp.exe */

#pragma once


EXTERN_C DECLSPEC_SELECTANY GUID HPXHeartBeatGuid =
   { 0x1178c091, 0x4a8d, 0x4657, 0xb6, 0x56, 0xce, 0x3, 0x0, 0x59, 0xc3, 0x4f };

EXTERN_C DECLSPEC_SELECTANY GUID QueueLengthGuid =
   { 0x9a7a620e, 0x19d0, 0x4697, 0xb6, 0xfa, 0xa8, 0x3, 0x84, 0x5d, 0x73, 0x29 };


EXTERN_C DECLSPEC_SELECTANY HANDLE HPXHeartBeat = NULL;

EXTERN_C DECLSPEC_SELECTANY struct {
    PERF_COUNTERSET_INFO CounterSet;
    PERF_COUNTER_INFO Counter0;
    PERF_COUNTER_INFO Counter1;
} QueueLengthInfo = {
    { { 0x9a7a620e, 0x19d0, 0x4697, 0xb6, 0xfa, 0xa8, 0x3, 0x84, 0x5d, 0x73, 0x29 },
    { 0x1178c091, 0x4a8d, 0x4657, 0xb6, 0x56, 0xce, 0x3, 0x0, 0x59, 0xc3, 0x4f },
    2, PERF_COUNTERSET_MULTI_AGGREGATE },
    { 1, PERF_COUNTER_RAWCOUNT, 0, sizeof(ULONG), PERF_DETAIL_NOVICE, 0, 0 },
    { 2, PERF_COUNTER_RAWCOUNT, 0, sizeof(ULONG), PERF_DETAIL_NOVICE, 0, 0 },
};

EXTERN_C FORCEINLINE
VOID
CounterCleanup(
    VOID
    )
{
    if (HPXHeartBeat != NULL) {
        PerfStopProvider(HPXHeartBeat);
        HPXHeartBeat = NULL;
    }
}

EXTERN_C FORCEINLINE
ULONG
CounterInitialize(
    __in_opt PERFLIBREQUEST NotificationCallback,
    __in_opt PERF_MEM_ALLOC MemoryAllocationFunction,
    __in_opt PERF_MEM_FREE MemoryFreeFunction,
    __inout_opt PVOID MemoryFunctionsContext
    )
{
    ULONG Status;
    PERF_PROVIDER_CONTEXT ProviderContext;

    ZeroMemory(&ProviderContext, sizeof(PERF_PROVIDER_CONTEXT));
    ProviderContext.ContextSize = sizeof(PERF_PROVIDER_CONTEXT);
    ProviderContext.ControlCallback = NotificationCallback;
    ProviderContext.MemAllocRoutine = MemoryAllocationFunction;
    ProviderContext.MemFreeRoutine = MemoryFreeFunction;
    ProviderContext.pMemContext = MemoryFunctionsContext;

    Status = PerfStartProviderEx(&HPXHeartBeatGuid,
                                 &ProviderContext,
                                 &HPXHeartBeat);
    if (Status != ERROR_SUCCESS) {
        HPXHeartBeat = NULL;
        return Status;
    }

    Status = PerfSetCounterSetInfo(HPXHeartBeat,
                                   &QueueLengthInfo.CounterSet,
                                   sizeof QueueLengthInfo);
    if (Status != ERROR_SUCCESS) {
        CounterCleanup();
        return Status;
    }
    return ERROR_SUCCESS;
}
