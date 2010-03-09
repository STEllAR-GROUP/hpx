#include <set>
#include <boost/array.hpp>
#include <boost/atomic.hpp>
#include <boost/thread.hpp>

template <typename int_type>
int_type generate_id(void)
{
    static boost::atomic<int_type> generator(0);
    return ++generator;
}

template <typename int_type, unsigned int buckets>
class static_hashed_set
{
public:
    bool insert(int_type const & id)
    {
        std::size_t index = id % buckets;

        boost::mutex::scoped_lock lock (ref_mutex[index]);

        std::pair<typename std::set<int_type>::iterator, bool> p;
        p = data[index].insert(id);

        return p.second;
    }

    bool find (int_type const & id)
    {
        std::size_t index = id % buckets;

        boost::mutex::scoped_lock lock (ref_mutex[index]);

        return data[index].find(id) != data[index].end();
    }

    bool erase(int_type const & id)
    {
        std::size_t index = id % buckets;

        boost::mutex::scoped_lock lock (ref_mutex[index]);

        if (data[index].find(id) != data[index].end())
        {
            data[index].erase(id);
            assert(data[index].find(id) == data[index].end());
            return true;
        }
        else
            return false;
    }

    int count_nodes(void) const
    {
        int ret = 0;
        for (int i = 0; i != buckets; ++i)
        {
            boost::mutex::scoped_lock lock (ref_mutex[i]);
            ret += data[i].size();
        }
        return ret;
    }

private:
    boost::array<std::set<int_type>, buckets> data;
    mutable boost::array<boost::mutex, buckets> ref_mutex;
};
