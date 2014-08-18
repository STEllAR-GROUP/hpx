#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_find.hpp>
#include <hpx/util/lightweight_test.hpp>

#include "test_utils.hpp"

void find_end_failing_test()
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, std::random_access_iterator_tag>
        decorated_iterator;

    std::vector<std::size_t> c(10007, 0);
    std::size_t h[] = {1,2};

    bool caught_exception = false;
    try {
        std::find_end(
            decorated_iterator(
                boost::begin(c),
                [](){ throw std::runtime_error("error"); }),
            decorated_iterator(boost::end(c)),
            boost::begin(h), boost::end(h));
        //should never reach this point
        HPX_TEST(false);
    }
    catch(hpx::exception_list const& e) {
        caught_exception = true;
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

int hpx_main()
{
    find_end_failing_test();
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency()));

    HPX_TEST_EQ_MSG(hpx::init(argc, argv, cfg), 0,
        "HPX main exted with non-zero status");

    return hpx::util::report_errors();
}
