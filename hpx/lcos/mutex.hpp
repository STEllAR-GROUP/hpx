//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_BARRIER_JUN_23_2008_0530PM)
#define HPX_LCOS_BARRIER_JUN_23_2008_0530PM

#include <boost/detail/atomic_count.hpp>
#include <boost/lockfree/fifo.hpp>

// Description of the mutex algorithm is explained here:
// http://lists.boost.org/Archives/boost/2006/09/110367.php
//
// The algorithm is: 
// 
// init(): 
//    active_count=0; 
//    no semaphore 
// 
// lock(): 
//    atomic increment active_count 
//    if new active_count ==1, that's us, so we've got the lock 
//    else 
//         get semaphore, and wait 
//         now we've got the lock 
// 
// unlock(): 
//    atomic decrement active_count 
//    if new active_count >0, then other threads are waiting, 
//        so release semaphore. 
// 
// locked(): 
//    return active_count>0 
// 
// get_semaphore(): 
//    if there's already a semaphore associated with this mutex, return that 
//    else 
//        create new semaphore. 
//        use atomic compare-and-swap to make this the associated semaphore if 
//            none 
//        if another thread beat us to it, and already set a semaphore, destroy 
//            new one, and return already-set one 
//        else return the new semaphore 

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos 
{
    // A mutex can be used to synchronize the access to a arbitrary resource
    class mutex 
    {
    public:
        mutex()
        {}

        void lock() {}
        void unlock() {}
        bool locked() { return false; }

    private:
    };

}}

#endif

