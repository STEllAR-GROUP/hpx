// Boost Logging library
//
// Author: John Torjo, www.torjo.com
//
// Copyright (C) 2007 John Torjo (see www.torjo.com for email)
//
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org for updates, documentation, and revision history.
// See http://www.torjo.com/log2/ for more details

namespace hpx { namespace util { namespace logging {

/**
@page after_destruction Using the logger(s)/filter(s) after they've been destroyed

- @ref after_destruction_can_happen
- @ref after_destruction_avoid
- @ref after_destruction_solution





\n\n
@section after_destruction_can_happen Can this happen?

The short answer : yes. How? The order of inialization between translation units is not defined, thus the same applies
for destruction. The following can happen:
- a global object is constructed @em before a logger
- thus, it will be destroyed @em after the logger
- if in its destructor it tries to use the logger, there we go - logger is used after it's been destroyed.



\n\n
@section after_destruction_avoid Avoiding the issue: making sure it never happens

Many thanks to Daniel Kruger for helping me with this:
- http://groups.google.ro/group/comp.lang.c++.moderated/tree/browse_frm/thread/17987673016b2098/d4c6bdcdca1e8fe9?hl=ro&rnum=1&_done=%2Fgroup%2Fcomp.lang.c%2B%2B.moderated%2Fbrowse_frm%2Fthread%2F17987673016b2098%3Fhl%3Dro%26#doc_aa38c20511f81615
- http://groups.google.ro/group/comp.lang.c++.moderated/tree/browse_frm/thread/17987673016b2098/d4c6bdcdca1e8fe9?hl=ro&rnum=1&_done=%2Fgroup%2Fcomp.lang.c%2B%2B.moderated%2Fbrowse_frm%2Fthread%2F17987673016b2098%3Fhl%3Dro%26#doc_f506c2b42f21dad9

The way to handle this is: since we can't handle initialization between translation units, we can handle initialization within the same translation unit.
In the same translation unit, if we have:
@code
static A a;
static B b;
@endcode

... we're guaranteed @c a is initialized before @c b. In other words, if in a translation unit @c a is defined before @c b,
@c a will be initialized before @c b.

Apply this to our problem:
- we just need some object that will reference the logger, to be defined before the global object that might end up using the logger in its destructor
  - note: referencing the logger will imply it gets constructed first
- this way, the logger will be created before the global object, and thus be destroyed after it

In every translation unit that has a global object that might end up using logger(s) on its destructor,
we need to create some object that will reference those loggers before the global object's definition.

Therefore, the amazing solution:

@code
// exposistion only
#define HPX_DECLARE_LOG(name,type) type* name (); \
    namespace { hpx::util::logging::ensure_early_log_creation ensure_log_is_created_before_main ## name ( * name () ); }
@endcode

When declaring the logger, we create a dummy static - in each translation unit -, that uses the logger; this will ensure the
logger is created before the global object that will use it.

@section after_destruction_solution The solution

All you need to do is :
- \#include the header file that declares the logs in all translation units that have global objects that could log from their destructor.
- of course, to be sure, you could \#include that header in all translation units :)

@note
This applies to filters as well.

*/

}}}
