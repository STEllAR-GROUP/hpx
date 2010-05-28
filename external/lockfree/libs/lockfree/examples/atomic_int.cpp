//  Copyright (C) 2009 Tim Blechmann
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/thread/thread.hpp>
#include <boost/lockfree/atomic_int.hpp>
#include <iostream>

boost::lockfree::atomic_int<int> counter(0);

const int iterations = 10000000;

void increment_counter(void)
{
    for (int i = 0; i != iterations; ++i)
        ++counter;
}

void decrement_counter(void)
{
    for (int i = 0; i != iterations; ++i)
        --counter;
}

int main(int argc, char* argv[])
{
    // incrementing counter in one thread, decrementing in another
    boost::thread thrd_inc(increment_counter);
    boost::thread thrd_dec(decrement_counter);

    thrd_inc.join();
    thrd_dec.join();

    std::cout << "counter value is " << counter << "." << std::endl;
}
