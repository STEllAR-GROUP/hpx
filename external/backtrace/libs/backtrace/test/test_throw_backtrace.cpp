#include <exception>
#include <boost/backtrace.hpp>
#include <iostream>

class my_exception : public std::exception, public boost::backtrace {
public:
};

int foo()
{
    throw my_exception();
    return 10;
}

int bar()
{
    return foo()+20;
}


int main()
{
    try {
        std::cout << bar() << std::endl;
    }
    catch(my_exception const &e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << boost::trace(e) << std::endl;
    }
}
