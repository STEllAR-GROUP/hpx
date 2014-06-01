#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/algorithm.hpp>
#include <hpx/util/lightweight_test.hpp>

#include "test_utils.hpp"

int hpx_main()
{
	std::cout << "test file testing";
	return hpx::finalize();
}

int main(int argc, char* argv[])
{
	std::vector<std::string cfg;
	cfg.push_back("hpx.os_threads=" +
		boost::lexical_cast<std::string>(hpx::threads::hardware::concurrency()));

	HPX_TEST_EQ_MSG(hpx::init(argc, argv, cfg), 0,
		"HPX main exited with non-zero status");

	return hpx::util::report_errors();

}