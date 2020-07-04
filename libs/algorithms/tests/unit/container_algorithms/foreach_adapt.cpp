#include <hpx/hpx_main.hpp>
#include <hpx/include/parallel_for_each.hpp>
#include <hpx/modules/testing.hpp>
#include "iter_sent.hpp"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <iostream>     

void myfunction (int i) {  
  std::cout << ' ' << i;
}

struct set_42
{
    template <typename T>
    void operator()(T& val)
    {
        val = T(42);
    }
};

int main()
{
     hpx::parallel::for_each(hpx::parallel::execution::seq,
         Iterator<std::int64_t>{0}, Sentinel<int64_t>{100}, myfunction);

    //HPX_TEST_EQ(result, std::int64_t(4950));

    //hpx::parallel::for_each(hpx::parallel::execution::seq,

    //)

    return hpx::util::report_errors();
}