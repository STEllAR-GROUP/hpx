//  Copyright (c) 2006, Giovanni P. Deretta
//
//  This code may be used under either of the following two licences:
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//  THE SOFTWARE. OF SUCH DAMAGE.
//
//  Or:
//
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)


#if !(defined(__i386__) || defined(__i486__) || defined(__i586__) || defined(__i686__))
#error This file is for x86 CPUs only.
#endif

#if !defined(__GNUC__)
#error This file requires gcc.
#endif

/*
 * This file should really be a plain assembler file.
 * Unfortunately Boost.Build v2 doesn't handle asm files (yet).
 * For now resort to inline asm.
 */


/*
   EAX is &from.sp
   EDX is to.sp
   This is the simplest version of swapcontext
   It saves registers on the old stack,
   saves the old stack pointer,
   load the new stack pointer,
   pop registers from the new stack
   and returns to new caller.
   EAX is simpy passed to the function it returns to.
   The first time EAX is the first parameter of the trampoline.
   Otherwise it is simply discarded.
   NOTE: This function should work on any IA32 CPU.
   NOTE: The biggest penalty is the last jump that
   will be always mis-predicted (~50 cycles on P4).
   We try to make its address available as soon as possible
   to try to reduce the penalty. Doing a ret instead of a
   'add $4, %esp'
   'jmp *%ecx'
   really kills performance.
   NOTE: popl is slightly better than mov+add to pop registers
   so is pushl rather than mov+sub.
   */

// Different systems interpret the specified alignment differently. Some
// interpret the number verbatim, others as the power of 2.

#if defined(__APPLE__)
#define HPX_COROUTINE_ALIGNMENT "4"
#define HPX_COROUTINE_TYPE_DIRECTIVE(name)
#else
#define HPX_COROUTINE_ALIGNMENT "16"
#define HPX_COROUTINE_TYPE_DIRECTIVE(name) ".type " #name ", @function\n\t"
#endif

#define HPX_COROUTINE_swapcontext(name)                                       \
    asm (                                                                     \
        ".text \n\t"                                                          \
        ".align " HPX_COROUTINE_ALIGNMENT " \n\t"                             \
        ".globl " #name "\n\t"                                                \
        HPX_COROUTINE_TYPE_DIRECTIVE(name)                                    \
    #name":\n\t"                                                              \
        "movl  16(%edx), %ecx\n\t"                                            \
        "pushl %ebp\n\t"                                                      \
        "pushl %ebx\n\t"                                                      \
        "pushl %esi\n\t"                                                      \
        "pushl %edi\n\t"                                                      \
        "movl  %esp, (%eax)\n\t"                                              \
        "movl  %edx, %esp\n\t"                                                \
        "popl  %edi\n\t"                                                      \
        "popl  %esi\n\t"                                                      \
        "popl  %ebx\n\t"                                                      \
        "popl  %ebp\n\t"                                                      \
        "add   $4, %esp\n\t"                                                  \
        "jmp   *%ecx\n\t"                                                     \
        "ud2\n\t"                                                             \
    )                                                                         \
/**/

HPX_COROUTINE_swapcontext(swapcontext_stack);
HPX_COROUTINE_swapcontext(swapcontext_stack2);
HPX_COROUTINE_swapcontext(swapcontext_stack3);

#undef HPX_COROUTINE_swapcontext
