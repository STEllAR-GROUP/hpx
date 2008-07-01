#include <climits>
#define BOOST_TEST_MODULE lockfree_tests
#include <boost/test/included/unit_test.hpp>

#include "test_helpers.hpp"

#include <boost/lockfree/stack.hpp>

#include <boost/thread.hpp>
using namespace boost;

const unsigned int buckets = 1<<10;
static_hashed_set<long, buckets> data;
boost::array<std::set<long>, buckets> returned;

boost::lockfree::stack<long> stk;

void add_items(void)
{
    unsigned long count = 1000000;

    for (long i = 0; i != count; ++i)
    {
        thread::yield();
        long id = generate_id<long>();

        bool inserted = data.insert(id);

        assert(inserted);

        stk.push(id);
    }
}

volatile bool running = true;

void get_items(void)
{
    for (;;)
    {
        thread::yield();
        long id;

        bool got = stk.pop(&id);
        if (got)
        {
            bool erased = data.erase(id);
            assert(erased);
        }
        else
            if (not running)
                return;
    }
}

BOOST_AUTO_TEST_CASE( stack_test )
{
    thread_group writer;
    thread_group reader;

    for (int i = 0; i != 2; ++i)
        reader.create_thread(&get_items);

    for (int i = 0; i != 2; ++i)
        writer.create_thread(&add_items);

    using namespace std;
    cout << "threads created" << endl;

    writer.join_all();

    cout << "writer threads joined, waiting for readers" << endl;

    running = false;
    reader.join_all();

    cout << "reader threads joined" << endl;

    BOOST_REQUIRE_EQUAL(data.count_nodes(), 0);
}
