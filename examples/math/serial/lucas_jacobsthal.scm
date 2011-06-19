;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;   Copyright (c) 2011 Bryce Lelbach
;
;   Distributed under the Boost Software License, Version 1.0. (See accompanying
;   file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define lucas-jacobsthal 
  (lambda (n)
    (cond
      ((= 0 n)  
        2)
      ((= 1 n)
        1)
      (else
        (+ (lucas-jacobsthal (- n 1)) (* 2 (lucas-jacobsthal (- n 2))))))))

(let ((n (string->number (list-ref (command-line) 1))))
  (display "lucas-jacobsthal(")
  (display n)
  (display ") == ")
  (display (lucas-jacobsthal n))
  (newline))

