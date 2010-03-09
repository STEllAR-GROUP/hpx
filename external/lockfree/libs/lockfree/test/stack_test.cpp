#include <climits>
#define BOOST_TEST_MODULE lockfree_tests
#include <boost/test/included/unit_test.hpp>

#include "test_helpers.hpp"

#include <boost/lockfree/stack.hpp>

#include <boost/thread.hpp>
using namespace boost;

template <typename freelist_t>
struct stack_tester
{
    static const unsigned int buckets = 1<<10;
    static const long node_count = 1000000;
    static const int reader_threads = 4;
    static const int writer_threads = 4;

    static_hashed_set<long, buckets> data;
    boost::array<std::set<long>, buckets> returned;

    atomic_int push_count, pop_count;

    boost::lockfree::stack<long, freelist_t> stk;

    stack_tester(void):
        push_count(0), pop_count(0)
    {}

    void add_items(void)
    {
        for (long i = 0; i != node_count; ++i)
        {
            long id = generate_id<long>();

            bool inserted = data.insert(id);
            assert(inserted);

            while(stk.push(id) == false)
                thread::yield();
            ++push_count;
        }
    }

    volatile bool running;

    void get_items(void)
    {
        for (;;)
        {
            long id;

            bool got = stk.pop(&id);
            if (got)
            {
                bool erased = data.erase(id);
                assert(erased);
                ++pop_count;
            }
            else
                if (not running)
                    return;
        }
    }

    void run(void)
    {
        BOOST_WARN(stk.is_lock_free());

        running = true;

        thread_group writer;
        thread_group reader;

        BOOST_REQUIRE(stk.empty());

        for (int i = 0; i != reader_threads; ++i)
            reader.create_thread(boost::bind(&stack_tester::get_items, this));

        for (int i = 0; i != writer_threads; ++i)
            writer.create_thread(boost::bind(&stack_tester::add_items, this));

        using namespace std;
        cout << "threads created" << endl;

        writer.join_all();

        cout << "writer threads joined, waiting for readers" << endl;

        running = false;
        reader.join_all();

        cout << "reader threads joined" << endl;

        BOOST_REQUIRE_EQUAL(data.count_nodes(), 0);
        BOOST_REQUIRE(stk.empty());

        BOOST_REQUIRE_EQUAL(push_count, pop_count);
        BOOST_REQUIRE_EQUAL(push_count, writer_threads * node_count);
    }
};


BOOST_AUTO_TEST_CASE( stack_test_caching )
{
    stack_tester<boost::lockfree::caching_freelist_t> tester;
    tester.run();
}

BOOST_AUTO_TEST_CASE( stack_test_static )
{
    stack_tester<boost::lockfree::static_freelist_t> tester;
    tester.run();
}
