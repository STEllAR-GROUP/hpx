;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;   Copyright (c) 2011 Bryce Lelbach
;
;   Distributed under the Boost Software License, Version 1.0. (See accompanying
;   file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define ackermann-peter 
  (lambda (m n)
    (if (= m 0)
      (+ n 1)
      (if (= n 0)
        (ackermann-peter (- m 1) 1)
        (ackermann-peter (- m 1) (ackermann-peter m (- n 1)))))))

(let ((m (string->number (list-ref (command-line) 1)))
      (n (string->number (list-ref (command-line) 2))))
  (display "ackermann-peter(")
  (display m)
  (display ", ")
  (display n)
  (display ") == ")
  (display (ackermann-peter m n))
  (newline))

