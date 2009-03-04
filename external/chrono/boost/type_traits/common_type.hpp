//  common_type.hpp  ---------------------------------------------------------//

//  Copyright 2008 Howard Hinnant
//  Copyright 2008 Beman Dawes

//  Distributed under the Boost Software License, Version 1.0.
//  See http://www.boost.org/LICENSE_1_0.txt

#ifndef BOOST_COMMON_TYPE_HPP
#define BOOST_COMMON_TYPE_HPP

#define BOOST_VARIADIC_COMMON_TYPE 0

#define BOOST_TYPEOF_SILENT
#include <boost/typeof/typeof.hpp>   // boost wonders never cease!

//----------------------------------------------------------------------------//
//                                                                            //
//                           C++03 implementation of                          //
//             20.6.7 Other transformations [meta.trans.other]                //
//                          Written by Howard Hinnant                         //
//                       Adapted for Boost by Beman Dawes                     //
//                                                                            //
//----------------------------------------------------------------------------//

namespace boost
{

  template <class T, class U = void, class V = void>
  struct common_type
  {
  public:
     typedef typename common_type<typename common_type<T, U>::type, V>::type type;
  };


  template <class T>
  struct common_type<T, void, void>
  {
  public:
     typedef T type;
  };

  template <class T, class U>
  struct common_type<T, U, void>
  {
  private:
     static T m_t();
     static U m_u();
     static bool m_f();  // workaround gcc bug; not required by std
  public:
     typedef BOOST_TYPEOF_TPL(m_f() ? m_t() : m_u()) type;
  };

}  // namespace boost

#endif  // BOOST_COMMON_TYPE_HPP
