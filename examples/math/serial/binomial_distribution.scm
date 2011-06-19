;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;   Copyright (c) 2011 Bryce Lelbach
;
;   Distributed under the Boost Software License, Version 1.0. (See accompanying
;   file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define factorial
  (lambda (n)
    (if (<= n 0) 1
      (* n (factorial (- n 1))))))

(define binomial-distribution
  (lambda (n p r)
    (let ((fn (factorial n))
          (fnr (factorial (- n r)))
          (fr (factorial r)))
      (* (/ fn (* fnr fr))
         (expt p r)
         (expt (- 1 p) (- n r))))))

(let ((n (string->number (list-ref (command-line) 1)))
      (p (string->number (list-ref (command-line) 2)))
      (r (string->number (list-ref (command-line) 3))))
  (display "binomial-distribution(")
  (display n)
  (display ", ")
  (display p)
  (display ", ")
  (display r)
  (display ") == ")
  (display (binomial-distribution n p r))
  (newline))

