////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <mutex>

struct BasicLockable {
    void lock() {}
    void unlock() {}
};

int main()
{
    BasicLockable basic_lockable;
    std::lock_guard<BasicLockable> lk(basic_lockable);

    std::lock_guard<BasicLockable> kl_adopt(basic_lockable, std::adopt_lock);
}
