#ifndef DEFS_HH
#define DEFS_HH

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
