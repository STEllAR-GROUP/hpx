//  (C) Copyright Giovanni P. Deretta 2005.
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

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
    NOTE: The biggest penality is the last jump that
    will be always mispredicted (~50 cycles on P4).
    We try to make its address available as soon as possible
    to try to reduce the penality. Doing a ret instead of a
    'add $4, %esp'
    'jmp *%ecx'
    really kills performance.
    NOTE: popl is slightly better than mov+add to pop registers
    so is pushl rather than mov+sub.
    */

    .text
    .align 16
    .globl swapcontext_stack,
    .type swapcontext_stack, @function

swapcontext_stack:
    movq  16(%rdx), %rcx
    pushq   %rbp
    pushq %rbx
    pushq %rsi
    pushq %rdi
    movq  %rsp, (%rax)
    movq  %rdx, %rsp
    popq  %rdi
    popq  %rsi
    popq  %rbx
    popq  %rbp
    add   $4, %rsp
//  popq %rcx
    jmp   *%rcx
    ud2

/*
   This is exactly the same than swapcontext_stack,
   but while the swapcontext_stack should be used
   for invocations, this should be used for yielding,
   thus there are two 'jmp' sites that, in the common
   invoke+yield case, each jump always to the same target
   and can be predicted (this is very important on P4).
   This optimization gives a 50% performance bonus on a plain
   'invoke and yield' test.
   NOTE: both subroutines work even if they are used in the
    wrong place.
  */

    .align 16
    .globl swapcontext_stack2,
    .type swapcontext_stack2, @function
swapcontext_stack2:
    movq  16(%rdx), %rcx
    pushq %rbp
    pushq %rbx
    pushq %rsi
    pushq %rdi
    movq  %rsp, (%rax)
    movq  %rdx, %rsp
    popq  %rdi
    popq  %rsi
    popq  %rbx
    popq  %rbp
    add   $4, %rsp
//  popq %rcx
    jmp   *%rcx
    ud2

    .align 16
    .globl swapcontext_stack3,
    .type swapcontext_stack3, @function
swapcontext_stack3:
    movq  16(%rdx), %rcx
    pushq %rbp
    pushq %rbx
    pushq %rsi
    pushq %rdi
    movq  %rsp, (%rax)
    movq  %rdx, %rsp
    popq  %rdi
    popq  %rsi
    popq  %rbx
    popq  %rbp
    add   $4, %rsp
//  popq  %rcx
    jmp   *%rcx
    ud2

    .align 16
    .globl swapcontext_stack_orig,
    .type swapcontext_stack_orig, @function
swapcontext_stack_orig:
    pushq %rbp
    pushq %rbx
    pushq %rsi
    pushq %rdi
    movq  %rsp, (%rax)
    movq  %rdx, %rsp
    popq  %rdi
    popq  %rsi
    popq  %rbx
    popq  %rbp
    ret
