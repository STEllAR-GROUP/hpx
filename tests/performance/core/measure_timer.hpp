#include <boost/progress.hpp>
#include <iostream>
#include <string>

class measure_timer
{  
public: 
    measure_timer(const char* str_): str(str_){}
    ~measure_timer();
private:
    boost::timer  t;
    std::string str;
};

measure_timer::~measure_timer()
{
    std::cout << str<< ":"<< t.elapsed() << std::endl << std::flush;
}
