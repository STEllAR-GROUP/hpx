// forward_constructor.hpp

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


#ifndef JT28092007_forward_constructor_HPP_DEFINED
#define JT28092007_forward_constructor_HPP_DEFINED

#if defined(HPX_MSVC) && (HPX_MSVC >= 1020)
# pragma once
#endif

#include <hpx/config.hpp>
#include <hpx/util/logging/detail/fwd.hpp>

#include <type_traits>

//#if BOOST_WORKAROUND(HPX_MSVC, BOOST_TESTED_AT(1400))

namespace hpx { namespace util { namespace logging {

#define HPX_LOGGING_FORWARD_CONSTRUCTOR(class_name,forward_to) \
        class_name() {} \
        template<class p1> class_name(const p1 & a1 ) : forward_to(a1) {} \
        template<class p1, class p2> class_name(const p1 & a1 , const p2 & a2)\
 : forward_to(a1,a2) {} \
        template<class p1, class p2, class p3> class_name\
(const p1 & a1 , const p2 & a2, const p3 & a3) : forward_to(a1,a2,a3) {} \
        template<class p1, class p2, class p3, class p4> class_name\
(const p1 & a1 , const p2 & a2, const p3 & a3, const p4 & a4)\
 : forward_to(a1,a2,a3,a4) {} \
        template<class p1, class p2, class p3, class p4, class p5> class_name\
(const p1 & a1 , const p2 & a2, const p3 & a3, const p4 & a4, const p5 & a5)\
 : forward_to(a1,a2,a3,a4,a5) {}

#define HPX_LOGGING_FORWARD_CONSTRUCTOR_INIT(class_name,forward_to,init) \
        class_name() { init (); } \
        template<class p1> class_name(const p1 & a1 ) : forward_to(a1) { init(); } \
        template<class p1, class p2> class_name(const p1 & a1 , const p2 & a2)\
 : forward_to(a1,a2) { init(); } \
        template<class p1, class p2, class p3> class_name\
(const p1 & a1 , const p2 & a2, const p3 & a3) : forward_to(a1,a2,a3) { init(); } \
        template<class p1, class p2, class p3, class p4> class_name\
(const p1 & a1 , const p2 & a2, const p3 & a3, const p4 & a4)\
 : forward_to(a1,a2,a3,a4) { init(); } \
        template<class p1, class p2, class p3, class p4, class p5>\
 class_name(const p1 & a1 , const p2 & a2, const p3 & a3, const p4 & a4, const p5 & a5)\
 : forward_to(a1,a2,a3,a4,a5) { init(); }

#ifdef HPX_MSVC
// workaround for VS - problem with copy constructor

#define HPX_LOGGING_FORWARD_CONSTRUCTOR_WITH_NEW(class_name,forward_to,type) \
        class_name() : forward_to(new type) {} \
        template<class p1> class_name(const p1 & a1 ) { \
            see_if_copy_constructor( a1, forward_to, \
                std::is_base_of<class_name,p1>() ); \
        } \
        template<class p1, class forward_type>\
 void see_if_copy_constructor(const p1 & a1, forward_type&, const std::true_type& )\
 { forward_to = a1.forward_to; \
        } \
        template<class p1, class forward_type> void see_if_copy_constructor\
(const p1 & a1, forward_type&, const std::false_type& ) { \
            forward_to = forward_type(new type(a1)); \
        } \
        template<class p1, class p2> class_name(const p1 & a1 , const p2 & a2)\
 : forward_to(new type(a1,a2)) {} \
        template<class p1, class p2, class p3>\
 class_name(const p1 & a1 , const p2 & a2, const p3 & a3)\
 : forward_to(new type(a1,a2,a3)) {} \
        template<class p1, class p2, class p3, class p4> class_name\
(const p1 & a1 , const p2 & a2, const p3 & a3, const p4 & a4)\
 : forward_to(new type(a1,a2,a3,a4)) {} \
        template<class p1, class p2, class p3, class p4, class p5>\
 class_name(const p1 & a1 , const p2 & a2, const p3 & a3, const p4 & a4, const p5 & a5)\
 : forward_to(new type(a1,a2,a3,a4,a5)) {}

#define HPX_LOGGING_FORWARD_CONSTRUCTOR_WITH_NEW_AND_INIT\
(class_name,forward_to,type, init) \
        class_name() : forward_to(new type) { init (); } \
        template<class p1> class_name(const p1 & a1 ) { \
            see_if_copy_constructor\
( a1, forward_to, std::is_base_of<class_name,p1>() ); \
        } \
        template<class p1, class forward_type>\
 void see_if_copy_constructor(const p1 & a1, forward_type&, const std::true_type& )\
 { forward_to = a1.forward_to; \
            init (); \
        } \
        template<class p1, class forward_type>\
 void see_if_copy_constructor(const p1 & a1, forward_type&, const std::false_type& )\
 { forward_to = forward_type(new type(a1)); \
            init (); \
        } \
        template<class p1, class p2> class_name(const p1 & a1 , const p2 & a2)\
 : forward_to(new type(a1,a2)) { init (); } \
        template<class p1, class p2, class p3> class_name\
(const p1 & a1 , const p2 & a2, const p3 & a3) : forward_to(new type(a1,a2,a3))\
 { init (); } \
        template<class p1, class p2, class p3, class p4> class_name\
(const p1 & a1 , const p2 & a2, const p3 & a3, const p4 & a4)\
 : forward_to(new type(a1,a2,a3,a4)) { init (); } \
        template<class p1, class p2, class p3, class p4, class p5> class_name\
(const p1 & a1 , const p2 & a2, const p3 & a3, const p4 & a4, const p5 & a5)\
 : forward_to(new type(a1,a2,a3,a4,a5)) { init (); }

#else
#define HPX_LOGGING_FORWARD_CONSTRUCTOR_WITH_NEW(class_name,forward_to,type) \
        class_name() : forward_to(new type) {} \
        template<class p1> class_name(const p1 & a1 ) : forward_to(new type(a1)) {} \
        template<class p1, class p2> class_name(const p1 & a1 , const p2 & a2)\
 : forward_to(new type(a1,a2)) {} \
        template<class p1, class p2, class p3> class_name\
(const p1 & a1 , const p2 & a2, const p3 & a3) : forward_to(new type(a1,a2,a3)) {} \
        template<class p1, class p2, class p3, class p4> class_name\
(const p1 & a1 , const p2 & a2, const p3 & a3, const p4 & a4)\
 : forward_to(new type(a1,a2,a3,a4)) {} \
        template<class p1, class p2, class p3, class p4, class p5>\
 class_name(const p1 & a1 , const p2 & a2, const p3 & a3, const p4 & a4, const p5 & a5)\
 : forward_to(new type(a1,a2,a3,a4,a5)) {}


#define HPX_LOGGING_FORWARD_CONSTRUCTOR_WITH_NEW_AND_INIT\
(class_name,forward_to,type, init) \
        class_name() : forward_to(new type) { init (); } \
        template<class p1> class_name(const p1 & a1 )\
 : forward_to(new type(a1)) { init (); } \
        template<class p1, class p2> class_name\
(const p1 & a1 , const p2 & a2) : forward_to(new type(a1,a2)) { init (); } \
        template<class p1, class p2, class p3> class_name\
(const p1 & a1 , const p2 & a2, const p3 & a3) : forward_to(new type(a1,a2,a3))\
 { init (); } \
        template<class p1, class p2, class p3, class p4> class_name\
(const p1 & a1 , const p2 & a2, const p3 & a3, const p4 & a4)\
 : forward_to(new type(a1,a2,a3,a4)) { init (); } \
        template<class p1, class p2, class p3, class p4, class p5>\
 class_name(const p1 & a1 , const p2 & a2, const p3 & a3, const p4 & a4, const p5 & a5)\
 : forward_to(new type(a1,a2,a3,a4,a5)) { init (); }

#endif

}}}

#endif

