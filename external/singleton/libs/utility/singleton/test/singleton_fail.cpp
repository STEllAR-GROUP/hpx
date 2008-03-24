/*=============================================================================
    Copyright (c) 2007 Tobias Schwinger
  
    Use modification and distribution are subject to the Boost Software 
    License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
    http://www.boost.org/LICENSE_1_0.txt).
==============================================================================*/

#include <boost/utility/singleton.hpp>

// test whether forgotten constructor is caught

struct X : boost::singleton<X>
{
};

int main()
{
  X x;

  return 0;
}
