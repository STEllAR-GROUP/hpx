
#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/atomic.hpp>

struct test
    : hpx::components::managed_component_base<test>
{
    test()
      : finished(false)
    {}
    test(bool finished_)
      : finished(finished_)
    {}

    ~test()
    {
        HPX_TEST(finished);
    }

    void pong()
    {}

    HPX_DEFINE_COMPONENT_ACTION(test, pong);

    void ping(hpx::id_type id, std::size_t iterations)
    {
        for(std::size_t i = 0; i != iterations; ++i)
        {
            pong_action()(id);
        }
        finished = true;
    }

    HPX_DEFINE_COMPONENT_ACTION(test, ping);

    boost::atomic<bool> finished;
};

typedef hpx::components::managed_component<test> test_component;
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(test_component);

HPX_REGISTER_ACTION(test::pong_action, test_pong_action);
HPX_REGISTER_ACTION(test::ping_action, test_ping_action);

int hpx_main(boost::program_options::variables_map & vm)
{
    {
        std::vector<hpx::id_type> localities = hpx::find_all_localities();
        hpx::id_type there = localities.size() == 1 ? localities[0] : localities[1];

        hpx::id_type id0 = hpx::components::new_<test>(localities[0]).get();
        hpx::id_type id1 = hpx::components::new_<test>(there, true).get();

        test::ping_action()(id0, id1, vm["iterations"].as<std::size_t>());
    }
    hpx::finalize();

    return hpx::util::report_errors();
}

int main(int argc, char **argv)
{
    boost::program_options::options_description desc(
        "usage: " HPX_APPLICATION_STRING " [options]");

    desc.add_options()
        ( "iterations",
          boost::program_options::value<boost::uint64_t>()->default_value(1000),
          "number of times to repeat the test")
        ;

    return hpx::init(desc, argc, argv);
}
