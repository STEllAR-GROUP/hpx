//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_algorithm.hpp>

#include <boost/range/functions.hpp>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
template <typename T>
struct safe_object
{
private:
    HPX_MOVABLE_ONLY(safe_object);

public:
    safe_object()
      : data_(hpx::get_os_thread_count())
    {
    }

    safe_object(safe_object && rhs)
      : data_(std::move(rhs.data_))
    {}

    safe_object& operator=(safe_object && rhs)
    {
        if (this != &rhs)
            data_ = std::move(rhs.data_);
        return *this;
    }

    T& get()
    {
        std::size_t idx = hpx::get_worker_thread_num();
        HPX_ASSERT(idx < hpx::get_os_thread_count());
        return data_[idx];
    }

    T const& get() const
    {
        std::size_t idx = hpx::get_worker_thread_num();
        HPX_ASSERT(idx < hpx::get_os_thread_count());
        return data_[idx];
    }

    template <typename F>
    void reduce (F const& f) const
    {
        for (T const& d : data_)
        {
            f(d);
        }
    }

private:
    std::vector<T> data_;
};

///////////////////////////////////////////////////////////////////////////////
std::vector<int> random_fill(std::size_t size)
{
    std::vector<int> c(size);
    std::generate(boost::begin(c), boost::end(c), std::rand);
    return c;
}

inline bool satisfies_criteria(int d)
{
    return d > 500 && (d % 7) == 0;
}

int hpx_main(int argc, char* argv[])
{
    using hpx::parallel::for_each;
    using hpx::parallel::par;

    // initialize data
    std::vector<int> data = random_fill(1000);

    // run a parallel loop to demonstrate thread safety of safe-object
    safe_object<std::vector<int> > ho;
    for_each(par, boost::begin(data), boost::end(data),
        [&ho](int d)
        {
            if (satisfies_criteria(d))
                ho.get().push_back(d);
        });

    // invoke the given reduce operation on the safe-object
    std::vector<int> result;
    ho.reduce(
        [&result](std::vector<int> const& chunk)
        {
            result.insert(result.end(), chunk.begin(), chunk.end());
        });

    // make sure all numbers conform to criteria
    for (int i : result)
    {
        if (!satisfies_criteria(i))
        {
            std::cout << "Number does not satisfy given criteria: " << i << "\n";
        }
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // Initialize and run HPX
    return hpx::init(argc, argv);
}
