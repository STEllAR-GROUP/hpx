;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;   Copyright (c) 2011 Bryce Lelbach
;
;   Distributed under the Boost Software License, Version 1.0. (See accompanying
;   file BOOST_LICENSE_1_0.rst or copy at http://www.boost.org/LICENSE_1_0.txt)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define pi 3.141592654)

(define generate-sin-values
  (lambda (i upper-bound step)
    (if (< i upper-bound)
      (begin
        (display i)
        (display ",")
        (display (sin i))
        (newline)
        (generate-sin-values (+ i step) upper-bound step)))))

(generate-sin-values (* -2 pi) (* 2 pi) (expt 10 -1))
  
