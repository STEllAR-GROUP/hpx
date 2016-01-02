//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This example demonstrates how a parallel::sort_by_key algorithm could be
// implemented based on the existing algorithm hpx::parallel::sort.

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_sort.hpp>
#include <hpx/include/iostreams.hpp>

#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace parallel
{
    struct extract_key
    {
        template <typename Tuple>
        auto operator() (Tuple && t) const
        ->  decltype(hpx::util::get<0>(std::forward<Tuple>(t)))
        {
            return hpx::util::get<0>(std::forward<Tuple>(t));
        }
    };

    template <typename KeyIter, typename ValueIter>
    void sort_by_key(KeyIter k_first, KeyIter k_last, ValueIter v_first)
    {
        typedef typename std::iterator_traits<KeyIter>::value_type key_value_type;

        ValueIter v_last = v_first;
        std::advance(v_last, std::distance(k_first, k_last));

        hpx::parallel::sort(
            hpx::parallel::par,
            hpx::util::make_zip_iterator(k_first, v_first),
            hpx::util::make_zip_iterator(k_last, v_last),
            std::less<key_value_type>(),
            extract_key());
    }
}

///////////////////////////////////////////////////////////////////////////////
void print_sequence(std::vector<int> const& keys, std::vector<char> const& values)
{
    for (std::size_t i = 0; i != keys.size(); ++i)
    {
        hpx::cout << "[" << keys[i] << ", " << values[i] << "]";
        if (i != keys.size() - 1)
            hpx::cout << ", ";
    }
    hpx::cout << std::endl;
}

int hpx_main()
{
    {
        std::vector<int> keys =    {  1,   4,   2,   8,   5,   7 };
        std::vector<char> values = { 'a', 'b', 'c', 'd', 'e', 'f' };

        hpx::cout << "unsorted sequence: {";
        print_sequence(keys, values);

        parallel::sort_by_key(keys.begin(), keys.end(), values.begin());

        hpx::cout << "sorted sequence:   {";
        print_sequence(keys, values);
    }

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    return hpx::init(argc, argv);   // Initialize and run HPX
}

