//////////////////////////////////////////////////////////////////////////////
//
// (C) Copyright David Abrahams, Vicente Botet, Ion Gaztanaga 2009.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org/libs/move for documentation.
//
//////////////////////////////////////////////////////////////////////////////

#include <boost/move/move.hpp>
#include <boost/container/vector.hpp>
#include "../example/movable.hpp"

int main()
{
   namespace bc = ::boost::container;
   //Default construct 10 movable objects
   bc::vector<movable> v(10);

   //Test default constructed value
   if(v[0].moved()){
      return 1;
   }

   //Move values
   bc::vector<movable> v2
      (boost::make_move_iterator(v.begin()), boost::make_move_iterator(v.end()));

   //Test values have been moved
   if(!v[0].moved()){
      return 1;
   }

   if(v2.size() != 10){
      return 1;
   }

   //Move again
   v.assign(boost::make_move_iterator(v2.begin()), boost::make_move_iterator(v2.end()));

   //Test values have been moved
   if(!v2[0].moved()){
      return 1;
   }

   if(v[0].moved()){
      return 1;
   }

   return 0;
}

/*
#include <boost/move/move.hpp>


class copy_movable
{
   BOOST_COPYABLE_AND_MOVABLE(copy_movable)
   int value_;

   public:
   copy_movable() : value_(1){}

   //Move constructor and assignment
   copy_movable(BOOST_RV_REF(copy_movable) m)
   {  value_ = m.value_;   m.value_ = 0;  }

   copy_movable(const copy_movable &m)
   {  value_ = m.value_;   }

   copy_movable & operator=(BOOST_RV_REF(copy_movable) m)
   {  value_ = m.value_;   m.value_ = 0;  return *this;  }

   copy_movable & operator=(BOOST_COPY_ASSIGN_REF(copy_movable) m)
   {  value_ = m.value_;   return *this;  }

   bool moved() const //Observer
   {  return value_ == 0; }
};

struct copy_movable_wrapper
{
   copy_movable cm;
};

copy_movable produce()
{  return copy_movable();  }


int main()
{
   copy_movable cm;
   cm = produce();
   
   const copy_movable_wrapper cmw;
   copy_movable_wrapper cmw2;
   cmw2 = cmw;

   return 0;
}
*/