////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2013 Shuangyang Yang
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_init.hpp>
#include <hpx/util/any.hpp>

#include <boost/serialization/access.hpp>

#include "small_big_object.hpp"

using boost::program_options::variables_map;
using boost::program_options::options_description;

using hpx::util::any;
using hpx::util::any_nonser;
using hpx::util::any_cast;

using hpx::init;
using hpx::finalize;

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    {
        // test equality
        {
            any any1(7), any2(7), any3(10), any4(std::string("seven"));

            std::cout << "-- equality of small object --" << std::endl;
            std::cout << "any1(7)==7:" << ((any1==7)?"true":"false") << std::endl;
            std::cout << "any1(7)==10:" << ((any1==10)?"true":"false") << std::endl;
            std::cout << "any1(7)==10.0f:" << ((any1==10.0f)?"true":"false") << std::endl;
            std::cout << "any1(7)==any1(7):" << ((any1==any1)?"true":"false") << std::endl;
            std::cout << "any1(7)==any2(7):" << ((any1==any2)?"true":"false") << std::endl;
            std::cout << "any1(7)==any3(10):" << ((any1==any3)?"true":"false") << std::endl;
            std::cout << "any1(7)==any4('seven'):" << ((any1==any4)?"true":"false") << std::endl;
            std::cout << std::endl;

            std::string long_str = 
                std::string("This is a looooooooooooooooooooooooooong string"); 
            std::string other_str = std::string("a different string");
            any1 = long_str;
            any2 = any1;
            any3 = other_str;
            any4 = 10.0f;


            std::cout << "-- equality of large object --" << std::endl;
            std::cout << "any1(LONG_STR)==LONG_STR:" << ((any1==long_str)?"true":"false") << std::endl;
            std::cout << "any1(LONG_STR)==OTHER_STR:" << ((any1==other_str)?"true":"false") << std::endl;
            std::cout << "any1(LONG_STR)==10.0f:" << ((any1==10.0f)?"true":"false") << std::endl;
            std::cout << "any1(LONG_STR)==any1(LONG_STR):" << ((any1==any1)?"true":"false") << std::endl;
            std::cout << "any1(LONG_STR)==any2(LONG_STR):" << ((any1==any2)?"true":"false") << std::endl;
            std::cout << "any1(LONG_STR)==any3(OTHER_STR):" << ((any1==any3)?"true":"false") << std::endl;
            std::cout << "any1(LONG_STR)==any4(10.0f):" << ((any1==any4)?"true":"false") << std::endl;
        }


        {
            if (sizeof(small_object) <= sizeof(void*))
                std::cout << "object is small\n";
            else
                std::cout << "object is large\n";

            small_object const f(17);

            any any1(f);
            any any2(any1);
            any any3;
            any3 = any1;

            (any_cast<small_object>(any1)) (7);
            (any_cast<small_object>(any2)) (9);
            (any_cast<small_object>(any3)) (11);
        }

        {
            if (sizeof(big_object) <= sizeof(void*))
                std::cout << "object is small\n";
            else
                std::cout << "object is large\n";

            big_object const f(5, 12);

            any any1(f);
            any any2(any1);
            any any3 = any1;

            (any_cast<big_object>(any1)) (0, 1);
            (any_cast<big_object>(any2)) (1, 0);
            (any_cast<big_object>(any3)) (1, 1);
        }
    }
    // non serializable version
    {
        {
            if (sizeof(small_object) <= sizeof(void*))
                std::cout << "object is small\n";
            else
                std::cout << "object is large\n";

            small_object const f(17);

            any_nonser any1_nonser(f);
            any_nonser any2_nonser(any1_nonser);
            any_nonser any3_nonser = any1_nonser;

            (any_cast<small_object>(any1_nonser)) (2);
            (any_cast<small_object>(any2_nonser)) (4);
            (any_cast<small_object>(any3_nonser)) (6);

        }

        {
            if (sizeof(big_object) <= sizeof(void*))
                std::cout << "object is small\n";
            else
                std::cout << "object is large\n";

            big_object const f(5, 12);

            any_nonser any1_nonser(f);
            any_nonser any2_nonser(any1_nonser);
            any_nonser any3_nonser = any1_nonser;

            (any_cast<big_object>(any1_nonser)) (3,4);
            (any_cast<big_object>(any2_nonser)) (5,6);
            (any_cast<big_object>(any3_nonser)) (7,8);
        }
    }

    finalize();

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description cmdline("Usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX
    return init(cmdline, argc, argv);
}

