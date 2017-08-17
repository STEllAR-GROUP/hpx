////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2017 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <exception>

struct exception {};

int main()
{
    std::exception_ptr p;

    // check make_exception_ptr
    {
        exception e;
        std::exception_ptr p = std::make_exception_ptr(e);
    }

    // check current_exception
    {
        try
        {
            throw exception();
        } catch (...) {
            std::exception_ptr p = std::current_exception();
        }
    }

    // check rethrow_exception
    {
        std::exception_ptr p;
        try
        {
            throw exception();
        } catch (...) {
            p = std::current_exception();
        }

        try
        {
            std::rethrow_exception(p);
        } catch (exception) {}
    }
}
