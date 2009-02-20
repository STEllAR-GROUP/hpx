//  Copyright (C) 2009 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  double tagged pointer, for ABA prevention in doubly linked lists

#if !defined(BOOST_DOUBLE_LOCKFREE_TAGGED_PTR_FEB_19_2009_0437PM)
#define BOOST_DOUBLE_LOCKFREE_TAGGED_PTR_FEB_19_2009_0437PM

#include <boost/lockfree/prefix.hpp>

#ifndef BOOST_LOCKFREE_PTR_COMPRESSION
#include <boost/lockfree/double_tagged_ptr_dcas.hpp>
#else
#include <boost/lockfree/double_tagged_ptr_ptrcompression.hpp>
#endif

#endif 
