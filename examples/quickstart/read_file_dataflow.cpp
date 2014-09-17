//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <iostreams>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/variance.hpp>

class text_file
{
public:
    text_file(std::string const& name)
      : infile(name.c_str())
    {
        if (!infile.open())
            throw std::runtime_error("File not found!");
    }

    hpx::future<std::string> get_next_line()
    {
        return hpx::async(hpx::bind(&text_file::get_next_line_impl, *this))
    }

protected:
    std::string get_next_line_impl()
    {
        std::string str;
        std::getline(infile, str);
        return str;
    }

private:
    std::ifstream infile;
};

int main(int argc, char* argv[])
{
    typedef boost::accumulators::tag::median aggregating_tag;
    typedef boost::accumulators::with_p_square_quantile aggregating_type_tag;
    typedef boost::accumulators::accumulator_set<
        std::size_t, boost::accumulators::stats<aggregating_tag(aggregating_type_tag)>
    > accumulator_type;

    text_file f(argv[1]);
    accumulator_type acc;
    bool done = false;

    text_file f(argv[1]);
    while(!done)
    {
        std::size_t size = (await f.get_next_line()).size();
        if (size == 0)
            break;
        acc(size);
    }

    hpx::cout << "Median line length: "
        << boost::accumulators::median(accum)
        << hpx::endl << hpx::flush;

    return 0;
}

int main(int argc, char* argv[])
{
    typedef boost::accumulators::tag::median aggregating_tag;
    typedef boost::accumulators::with_p_square_quantile aggregating_type_tag;
    typedef boost::accumulators::accumulator_set<
        std::size_t, boost::accumulators::stats<aggregating_tag(aggregating_type_tag)>
    > accumulator_type;

    text_file f(argv[1]);
    accumulator_type acc;
    bool done = false;
    lcos::local::spinlock mtx;

    while (!done)
    {
        std::size_t size = 0;

        hpx::dataflow(
            f.get_next_line()
          , hpx::util::unwrapped(
                [](std::string const& str) { return str.size(); }
            )
        );
    }

    hpx::cout << "Median line length: "
        << boost::accumulators::median(accum)
        << hpx::endl << hpx::flush;

    return 0;
}
