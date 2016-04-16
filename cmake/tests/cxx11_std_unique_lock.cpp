////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <mutex>
#include <utility>

struct Lockable {
    void lock() {}
    bool try_lock() { return true; }
    void unlock() {}
};

int main()
{
    Lockable lockable;
    std::unique_lock<Lockable> lk(lockable);
    lk.lock();
    lk.try_lock();
    lk.owns_lock();
    lk.unlock();
    lk.release();

    std::unique_lock<Lockable> mlk(std::move(lk));

    std::unique_lock<Lockable> kl_adopt(lockable, std::adopt_lock);
    std::unique_lock<Lockable> kl_defer(lockable, std::defer_lock);
    std::unique_lock<Lockable> kl_try(lockable, std::try_to_lock);
}
