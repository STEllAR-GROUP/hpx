// Copyright (c) 2013 Erik Schnetter
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef DEFS_HH
#define DEFS_HH

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>



template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T> v)
{
  os << "[";
  for (std::size_t i=0; i<v.size(); ++i) {
    if (i != 0) os << ",";
    os << v[i];
  }
  os << "]";
  return os;
}

template<typename T>
std::string mkstr(const T& x)
{
  std::ostringstream os;
  os << x;
  return os.str();
}

#endif  // #ifndef DEFS_HH
