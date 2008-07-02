;  Copyright (c) 2007-2008 Hartmut Kaiser
;
;  Distributed under the Boost Software License, Version 1.0.
;  (See accompanying file LICENSE_1_0.txt or copy at
;  http://www.boost.org/LICENSE_1_0.txt)

PUBLIC swapcontext_stack
PUBLIC swapcontext_stack2
PUBLIC swapcontext_stack3

;   RCX is &from.sp	
;   RDX is to.sp
;
;   This simple version of swapcontext_stack saves registers 
;   on the old stack, saves the old stack pointer, loads the 
;   new stack pointer, pop registers from the new stack
;   and returns to the new caller.
;
;   NOTE: This function should work on any IA64 CPU.
;   NOTE: The biggest penalty is the last jump that
;         will be always mis-predicted (~50 cycles on P4).
;
;   RCX is simply passed to the function it returns to. The first 
;   time RCX is the first parameter of the trampoline. Otherwise 
;   it is simply discarded.
;
;   We try to make its address available as soon as possible
;   to try to reduce the penalty. Doing a return instead of a
;
;      add rsp, 8'
;      jmp rax
;
;   really kills performance.
;
;   NOTE: popl is slightly better than mov+add to pop registers
;         so is pushl rather than mov+sub.
;

_TEXT	SEGMENT
swapcontext_stack proc
;
    mov rax, qword ptr [rdx+32]
    push rbp
    push rbx
    push rsi
    push rdi
    mov [rcx], rsp
    mov rsp, rdx
    pop rdi
    pop rsi
    pop rbx
    pop rbp
    add rsp, 8
    mov rcx, qword ptr [rdx+48]
    jmp rax
;
swapcontext_stack endp
;
swapcontext_stack2 proc
;
    mov rax, qword ptr [rdx+32]
    push rbp
    push rbx
    push rsi
    push rdi
    mov [rcx], rsp
    mov rsp, rdx
    pop rdi
    pop rsi
    pop rbx
    pop rbp
    add rsp, 8
    mov rcx, qword ptr [rdx+48]
    jmp rax
;
swapcontext_stack2 endp
;
swapcontext_stack3 proc
;
    mov rax, qword ptr [rdx+32]
    push rbp
    push rbx
    push rsi
    push rdi
    mov [rcx], rsp
    mov rsp, rdx
    pop rdi
    pop rsi
    pop rbx
    pop rbp
    add rsp, 8
    mov rcx, qword ptr [rdx+48]
    jmp rax
;
swapcontext_stack3 endp
_TEXT ends

    end
    