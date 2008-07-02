;  Copyright (c) 2007-2008 Hartmut Kaiser
;
;  Distributed under the Boost Software License, Version 1.0.
;  (See accompanying file LICENSE_1_0.txt or copy at
;  http://www.boost.org/LICENSE_1_0.txt)

PUBLIC CAS2_windows64

_TEXT	SEGMENT
CAS2_windows64 proc
;
    mov rax, qword ptr [rsp+28h] 
    mov rdx, r9 
    mov rbx, r8
    mov rdi, rcx
    mov rcx, rdx 
    lock cmpxchg16b qword ptr [rdi]
    xor rax, rax
    ret
;
CAS2_windows64 endp
_TEXT ends

    end

