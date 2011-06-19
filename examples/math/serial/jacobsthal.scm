;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;   Copyright (c) 2011 Bryce Lelbach
;
;   Distributed under the Boost Software License, Version 1.0. (See accompanying
;   file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define jacobsthal 
  (lambda (n)
    (cond
      ((= 0 n)  
        0)
      ((= 1 n)
        1)
      (else
        (+ (jacobsthal (- n 1)) (* 2 (jacobsthal (- n 2))))))))

(let ((n (string->number (list-ref (command-line) 1))))
  (display "jacobsthal(")
  (display n)
  (display ") == ")
  (display (jacobsthal n))
  (newline))

