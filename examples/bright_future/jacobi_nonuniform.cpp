//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#ifdef BRIGHT_FUTURE_NO_HPX
#include <iostream>
#else
#include <hpx/include/iostreams.hpp>
#endif

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <algorithm>

#include "sparse_matrix.hpp"

#include <boost/spirit/include/qi_numeric.hpp>
#include <boost/spirit/include/qi_char.hpp>
#include <boost/spirit/include/qi_string.hpp>
#include <boost/spirit/include/qi_action.hpp>
#include <boost/spirit/include/qi_parse.hpp>
#include <boost/spirit/include/qi_auxiliary.hpp>
#include <boost/spirit/include/qi_operator.hpp>
#include <boost/spirit/include/support_istream_iterator.hpp>
#include <boost/spirit/include/phoenix.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

#ifdef BRIGHT_FUTURE_NO_HPX
using std::cout;
using std::flush;
#else
using hpx::init;
using hpx::finalize;

using hpx::cout;
using hpx::flush;
#endif

namespace qi = boost::spirit::qi;
namespace phx = boost::phoenix;

extern void solve(
    bright_future::crs_matrix<double> const & A
  , std::vector<double> & x
  , std::vector<double> const & b
  , std::size_t
  , std::size_t
);

bool non_zero(double d)
{
    return std::abs(d) > 1e-20;
}

void add_entry(bright_future::crs_matrix<double> & M, double d, std::size_t & index, std::size_t & n_row)
{
    if(non_zero(d))
    {
        M.values.push_back(d);
        M.indices.push_back(index);
        ++n_row;
    }
    ++index;
}

void new_row(bright_future::crs_matrix<double> & M, std::size_t & n_row, std::size_t & index)
{
    M.rows.push_back(M.rows.back() + n_row);
    n_row = 0;
    index = 0;
}

void init_mtx(bright_future::crs_matrix<double> & A, std::size_t & n_row, std::size_t & non_zeros)
{
    A.values.reserve(non_zeros);
    A.indices.reserve(non_zeros);
    A.rows.reserve(n_row + 1);
    A.rows.push_back(0);
}

void add_entry_mtx(bright_future::crs_matrix<double> & A, std::size_t & cur_row, std::size_t & cur_n, std::size_t j, std::size_t i, double v)
{
    A.values.push_back(v);
    A.indices.push_back(j);
    ++cur_n;
    if(i != cur_row)
    {
        cur_row = i;
        A.rows.push_back(A.rows.back() + cur_n);
        cur_n = 0;
    }
}

int hpx_main(variables_map & vm)
{
    {
        std::size_t max_iterations  = vm["max_iterations"].as<std::size_t>();
        std::size_t block_size      = vm["block_size"].as<std::size_t>();
        std::string matrix_name     = vm["matrix"].as<std::string>();
        std::string vector_name     = vm["vector"].as<std::string>();
        std::string input_format    = vm["format"].as<std::string>();
        std::string mode            = vm["mode"].as<std::string>();

        bright_future::crs_matrix<double> A;
        std::vector<double> x;
        std::vector<double> b;
        if(input_format == "mtx")
        {
            std::ifstream file(matrix_name.c_str());
            file.unsetf(std::ios_base::skipws);
            typedef boost::spirit::istream_iterator iterator;
            iterator begin(file);
            iterator end;

            std::size_t current_row = 0;
            std::size_t nz_row = 0;
            std::size_t dim = 0;
            std::size_t nz_entries = 0;

            qi::phrase_parse(
                begin
              , end
              , (qi::int_[phx::ref(dim) = qi::_1] >> qi::int_ >> qi::int_[phx::ref(nz_entries) = qi::_1])
                [phx::bind(init_mtx, phx::ref(A), phx::ref(dim), phx::ref(nz_entries))]
                >>
                *(  // entry
                    (qi::int_ >> qi::int_ >> qi::double_)
                    [phx::bind(add_entry_mtx, phx::ref(A), phx::ref(current_row), phx::ref(nz_row), qi::_1, qi::_2, qi::_3)]
                )
              , qi::space
            );

            std::cout << "A: " << dim << "x" << dim << " number of non zeros: " << nz_entries << "\n";
            x = std::vector<double>(dim, 1.0);
            b = std::vector<double>(dim, 1.0);
        }
        else if(input_format == "matlab")
        {
            bright_future::crs_matrix<double> A;
            {
                std::ifstream file(matrix_name.c_str());
                file.unsetf(std::ios_base::skipws);
                typedef boost::spirit::istream_iterator iterator;
                iterator begin(file);
                iterator end;

                A.rows.push_back(0);
                std::size_t idx = 0;
                std::size_t row = 0;
                std::size_t nz_row = 0;

                qi::phrase_parse(
                    begin
                  , end
                  , *(  // column:
                        *(
                            qi::double_
                            [
                                phx::bind(add_entry, phx::ref(A), qi::_1, phx::ref(idx), phx::ref(nz_row))
                            ] >> -qi::lit(",")
                        )
                     >> qi::eol
                        [
                            phx::bind(new_row, phx::ref(A), phx::ref(nz_row), phx::ref(idx))
                        ]
                    )
                  , +qi::lit(" ")
                );
                std::cout << "A: " << row << "x" << row << " nonzero values: " << A.values.size() << "\n";
            }
            {
                std::ifstream file(vector_name.c_str());
                file.unsetf(std::ios_base::skipws);
                typedef boost::spirit::istream_iterator iterator;
                iterator begin(file);
                iterator end;
                qi::phrase_parse(begin, end, *qi::double_, qi::ascii::space, b);
            }

            std::cout << "b: " << b.size() << "\n";
            x = std::vector<double>(b.size(), 1.0);
        }

        if(mode == "solve")
        {
            solve(A,x,b, block_size, max_iterations);
        }
        else if(mode == "statistics")
        {
            std::size_t min_per_row = (std::numeric_limits<std::size_t>::max)();
            std::size_t max_per_row = 0;
            double mean_per_row = 0.0;
            for(std::size_t r = 0; r < b.size(); ++r)
            {
                const std::size_t begin = A.row_begin(r);
                const std::size_t end = A.row_end(r);
                std::size_t n_row = end-begin;
                mean_per_row += n_row;
                if(n_row > max_per_row)
                {
                    max_per_row = n_row;
                }
                if(n_row < min_per_row)
                {
                    min_per_row = n_row;
                }
                /*
                for(std::size_t c = begin; c < end; ++c)
                {
                }
                */
            }
            std::cout << "Matrix has " << A.values.size() << " non zero entries\n";
            std::cout << "order: " << b.size() << "x" << b.size() << "\n";
            std::cout << "Entries per row:\n";
            std::cout << "\tmax " << max_per_row << "\n";
            std::cout << "\tmin " << min_per_row << "\n";
            std::cout << "\tmean " << mean_per_row/b.size() << "\n";
            std::cout << "Density is: " << static_cast<double>(A.values.size())/(b.size() * b.size()) << "\n";
        }

#ifndef BRIGHT_FUTURE_NO_HPX
        finalize();
#endif
    }
    return 0;
}

int main(int argc, char **argv)
{
    options_description
        desc_commandline("usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        (
            "format"
          , value<std::string>()->default_value("matlab")
          , "input format"
        )
        (
            "mode"
          , value<std::string>()->default_value("matlab")
          , "input format"
        )
        (
            "matrix"
          , value<std::string>()->default_value("matrix.txt")
          , "matrix file"
        )
        (
            "vector"
          , value<std::string>()->default_value("vector.txt")
          , "vector file"
        )
        (
            "n"
          , value<std::size_t>()->default_value(10)
          , "number of points"
        )
        (
            "seed"
          , value<std::size_t>()->default_value(10)
          , "seed"
        )
        (
            "max_iterations"
          , value<std::size_t>()->default_value(10)
          , "Maximum number of iterations"
        )
        (
            "block_size"
          , value<std::size_t>()->default_value(1000)
          , "How to block the iteration"
        )
        ;

#ifdef BRIGHT_FUTURE_NO_HPX
    variables_map vm;

    desc_commandline.add_options()
        (
            "help", "This help message"
        )
        ;

    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc_commandline), vm);

    if(vm.count("help"))
    {
        cout << desc_commandline;
        return 0;
    }

    return hpx_main(vm);
#else
    return init(desc_commandline, argc, argv);
#endif
}
