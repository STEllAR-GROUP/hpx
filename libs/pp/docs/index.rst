..
    Copyright (c) 2018 The STE||AR-Group

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _libs_pp:

===========
pp
===========

This library contains useful preprocessor macros:

.. c:function:: HPX_PP_CAT(A, B)

Concatenates the tokens `A` and `B` into `AB`

.. c:function HPX_PP_EXPANDS(x)

Expands the preprocessor token `x`

.. c:function HPX_PP_NARGS(...)

Determines the number of arguments passed to a variadic macro

.. c:function HPX_PP_STRINGIZE(A)

Turns the token `A` into the string literal `"A"`

.. c:function HPX_PP_STRIP_PARENS(A)

Strips parenthesis from the token A. For example, `HPX_PP_STRIP_PARENS((A))`
results in `A`
