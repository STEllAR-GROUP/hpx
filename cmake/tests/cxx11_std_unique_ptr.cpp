////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2001 John Maddock
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <memory>

template <typename T>
struct ptr
{
private:
    T* p;

public:
    ptr(std::unique_ptr<T>& r)
    {
        p = r.release();
    }

    ptr& operator=(std::unique_ptr<T>& r)
    {
        delete p;
        p = r.release();
        return *this;
    }

    ~ptr()
    {
        delete p;
    }
};

int main()
{
    std::unique_ptr<int> up1(new int);
    ptr<int> mp(up1);
    std::unique_ptr<int> up2(new int);
    mp = up2;
}
