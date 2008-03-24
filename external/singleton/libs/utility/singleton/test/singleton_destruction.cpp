/*=============================================================================
    Copyright (c) 2007 Tobias Schwinger
  
    Use modification and distribution are subject to the Boost Software 
    License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
    http://www.boost.org/LICENSE_1_0.txt).
==============================================================================*/

#include <boost/utility/singleton.hpp>
#include <boost/utility/mutexed_singleton.hpp>
#include <boost/utility/thread_specific_singleton.hpp>

#include <boost/detail/lightweight_test.hpp>

template< int DisposalSlot >
struct counter
{
    static int n;
};

template< int DisposalSlot > int counter<DisposalSlot>::n = 0;

int prev_dtor = 0;

template< int Id, int DisposalSlot >
class X
  : public boost::singleton< X<Id,DisposalSlot>, DisposalSlot >
{
    int i;
  public:

    X(boost::restricted)
      : i(DisposalSlot*100 + 10 - ++counter<DisposalSlot>::n) 
    { }

    ~X() 
    { 
        BOOST_TEST(i > prev_dtor); 
        prev_dtor = i;
    }
};

template< int Id, int DisposalSlot >
class Y
  : public boost::mutexed_singleton< Y<Id,DisposalSlot>, DisposalSlot >
{
    int i;
  public:

    Y(boost::restricted)
      : i(DisposalSlot*100 + 10 - ++counter<DisposalSlot>::n) 
    { }

    ~Y() 
    { 
        BOOST_TEST(i > prev_dtor); 
        prev_dtor = i;
    }
};

template< int Id, int DisposalSlot >
class Z
  : public boost::thread_specific_singleton< Z<Id,DisposalSlot>, DisposalSlot >
{
    int i;
  public:

    Z(boost::restricted)
      : i(DisposalSlot*100 + 10 - ++counter<DisposalSlot>::n) 
    { }

    ~Z() 
    { 
        BOOST_TEST(i > prev_dtor); 
        prev_dtor = i;
    }
};

int main()
{
    X<1,0>::instance.operator->();
    Y<1,0>::instance.operator->();
    Z<1,0>::instance.operator->();
    X<1,1>::instance.operator->();
    Y<1,1>::instance.operator->();
    Z<1,1>::instance.operator->();
    X<1,2>::instance.operator->();
    Y<1,2>::instance.operator->();
    Z<1,2>::instance.operator->();
    X<0,1>::instance.operator->();
    Y<0,1>::instance.operator->();
    Z<0,1>::instance.operator->();
    X<1,1>::instance.operator->();
    Y<1,1>::instance.operator->();
    Z<1,1>::instance.operator->();
    X<2,1>::instance.operator->();
    Y<2,1>::instance.operator->();
    Z<2,1>::instance.operator->();

    boost::destroy_singletons();

    return boost::report_errors();
}

