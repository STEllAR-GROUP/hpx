//  Copyright (c) 2007 Robert Perricone
//  Copyright (c) 2007-2008 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#if !defined(__x86_64__)
#error This file is for x86 CPUs only.
#endif

#if !defined(__GNUC__)
#error This file requires compilation with gcc.
#endif

/*
 *  This file should really be a plain assembler file.
 *  Unfortunately Boost.Build v2 doesn't handle asm files (yet).
 *  For now resort to inline asm.
 */

//     RDI is &from.sp	
//     RSI is to.sp
//
//     This is the simplest version of swapcontext
//     It saves registers on the old stack, saves the old stack pointer,
//     load the new stack pointer, pop registers from the new stack
//     and returns to new caller.
//
//     RDI is set to be the parameter for the function to be called.
//     The first time RDI is the first parameter of the trampoline.
//     Otherwise it is simply discarded.
//
//     NOTE: This function should work on any IA64 CPU.
//     NOTE: The biggest penalty is the last jump that
//           will be always mis-predicted (~50 cycles on P4).
//
//     We try to make its address available as soon as possible
//     to try to reduce the penalty. Doing a return instead of a
//
//        'add $8, %esp'
//        'jmp *%ecx'
//
//     really kills performance.
//
//     NOTE: popl is slightly better than mov+add to pop registers
//           so is pushl rather than mov+sub.

#define BOOST_COROUTINE_SWAPCONTEXT(name)                                     \
    asm volatile (                                                            \
        ".text \n\t"                                                          \
        ".global " #name "\n\t"                                               \
        ".type " #name ", @function\n\t"                                      \
        ".align 16\n"                                                         \
    #name ":\n\t"                                                             \
        "movq  32(%rsi), %rcx\n\t"                                            \
        "pushq %rbp\n\t"                                                      \
        "pushq %rbx\n\t"                                                      \
        "pushq %rax\n\t"                                                      \
        "pushq %rdx\n\t"                                                      \
        "movq  %rsp, (%rdi)\n\t"                                              \
        "movq  %rsi, %rsp\n\t"                                                \
        "popq  %rdx\n\t"                                                      \
        "popq  %rax\n\t"                                                      \
        "popq  %rbx\n\t"                                                      \
        "popq  %rbp\n\t"                                                      \
        "movq 48(%rsi), %rdi\n\t"                                             \
        "add   $8, %rsp\n\t"                                                  \
        "jmp   *%rcx\n\t"                                                     \
        "ud2\n\t"                                                             \
    )                                                                         \
/**/

BOOST_COROUTINE_SWAPCONTEXT(swapcontext_stack);
BOOST_COROUTINE_SWAPCONTEXT(swapcontext_stack2);
BOOST_COROUTINE_SWAPCONTEXT(swapcontext_stack3);

#undef BOOST_COROUTINE_SWAPCONTEXT

