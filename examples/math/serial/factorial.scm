;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;   Copyright (c) 2011 Bryce Lelbach
;   Copyright (c) 2001-2011 Joel de Guzman
;   Copyright (c) 2001-2011 Hartmut Kaiser
;
;   Distributed under the Boost Software License, Version 1.0. (See accompanying
;   file BOOST_LICENSE_1_0.rst or copy at http://www.boost.org/LICENSE_1_0.txt)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define factorial
  (lambda (n)
    (if (<= n 0) 1
      (* n (factorial (- n 1))))))

(let ((n (string->number (list-ref (command-line) 1))))
  (display "factorial(")
  (display n)
  (display ") == ")
  (display (factorial n))
  (newline))

