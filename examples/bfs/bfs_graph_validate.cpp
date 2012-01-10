//  Copyright (c) 2011 Matthew Anderson
//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

#include <boost/dynamic_bitset.hpp>

#include "bfs_graph_validate.hpp"
#include "bfs_graph_util.hpp"

namespace bfs_graph
{
    ///////////////////////////////////////////////////////////////////////////
    // this routine mirrors the matlab validation routine
    int validate_graph(std::size_t searchkey,
        std::vector<std::size_t> const& parents,
        std::vector<std::pair<std::size_t, std::size_t> > const& edgelist,
        std::size_t& num_nodes)
    {
        // find the largest edge number
        // octave:  N = max (max (ij));
        std::size_t N = std::accumulate(edgelist.begin(), edgelist.end(),
            std::size_t(0), bfs_graph::max_node) + 1;

        // Get the number of edges for performance counting
        std::vector<std::size_t> nedge_bins(N, 0);
        for (std::size_t i = 0; i < edgelist.size(); ++i) {
            ++nedge_bins[edgelist[i].first];
            ++nedge_bins[edgelist[i].second];
        }

        // Volume/2
        num_nodes = std::accumulate(
            nedge_bins.begin(), nedge_bins.end(), std::size_t(0)) / 2;

        // start validation
        if (parents[searchkey] != searchkey) {
            // the parent of the searchkey is always itself
            std::cout << "Invalid parent for searchkey: " << searchkey
                      << " (parent: " << parents[searchkey] << ")" << std::endl;
            return -1;
        }

        // Find the indices of the node-parents list that are nonzero
        // octave: level = zeros (size (parent));
        // octave: level (slice) = 1;
        std::vector<std::size_t> slice, level(parents.size());
        for (std::size_t i = 0; i < parents.size(); ++i) {
            if (parents[i] > 0) {
                slice.push_back(i);
                level[i] = 1;
            }
            else {
                level[i] = 0;
            }
        }

        // Define a mask
        // octave: P = parent (slice);
        // octave:  mask = P != search_key;
        std::vector<std::size_t> P(slice.size());
        boost::dynamic_bitset<> mask(slice.size());
        for (std::size_t i = 0; i < slice.size(); ++i) {
            P[i] = parents[slice[i]];
            mask[i] = P[i] != searchkey;
        }

        // octave:  while any (mask)
        std::size_t k = 0;
        while (mask.any()) {
            // octave:  level(slice(mask)) = level(slice(mask)) + 1;
            for (std::size_t i = 0; i < slice.size(); ++i) {
                if (mask[i])
                    ++level[slice[i]];
            }

            // octave:  P = parent(P)
            for (std::size_t i = 0; i < P.size(); ++i) {
                P[i] = parents[P[i]];
                mask[i] = P[i] != searchkey;
            }

            if (++k > N)     // there is a cycle in the tree -- something wrong
                return -3;
        }

        // octave: lij = level (ij);
        std::size_t num_edges = edgelist.size();
        std::vector<std::size_t> li(num_edges), lj(num_edges);
        for (std::size_t i = 0; i < num_edges; ++i) {
            li[i] = level[edgelist[i].first];
            lj[i] = level[edgelist[i].second];
        }

        // octave: neither_in = lij(1,:) == 0 & lij(2,:) == 0;
        //         both_in = lij(1,:) > 0 & lij(2,:) > 0;
        boost::dynamic_bitset<> neither_in(num_edges), both_in(num_edges);
        for (std::size_t i = 0; i < num_edges;i++) {
            if (li[i] == 0 && lj[i] == 0)
                neither_in[i] = true;
            if (li[i] > 0 && lj[i] > 0)
                both_in[i] = true;
        }

        // octave:
        //  if any (not (neither_in | both_in)),
        //    out = -4;
        //    return
        for (std::size_t i = 0; i < num_edges; ++i) {
            if (!(neither_in[i] || both_in[i]))
                return -4;
        }

        // octave: respects_tree_level = abs (lij(1,:) - lij(2,:)) <= 1;
        boost::dynamic_bitset<> respects_tree_level(num_edges);
        for (std::size_t i = 0; i < num_edges; ++i)
            respects_tree_level[i] = abs((int)(li[i] - lj[i])) <= 1;

        // octave:
        // if any (not (neither_in | respects_tree_level)),
        //  out = -5;
        //  return
        for (std::size_t i = 0; i < num_edges; ++i) {
            if (!(neither_in[i] || respects_tree_level[i]))
                return -5;
        }

        return 1;
    }

    ///////////////////////////////////////////////////////////////////////////
    struct stddev
    {
        stddev(double mean) : mean_(mean) {}

        template <typename T>
        double operator()(double curr, T x) const
        {
            return curr + (x - mean_) * (x - mean_);
        }

        double mean_;
    };

    template <typename T>
    void get_statistics(std::vector<T> x,
        double &minimum, double &mean, double &deviation, double &firstquartile,
        double &median, double &thirdquartile, double &maximum)
    {
        // Compute mean
        std::size_t n = x.size();
        mean = std::accumulate(x.begin(), x.end(), T(0)) / double(n);

        // Compute std deviation
        deviation = std::accumulate(x.begin(), x.end(), 0.0, stddev(mean)) /
            std::sqrt(n - 1.0);

        // Sort x
        std::sort(x.begin(), x.end());

        minimum = double(x[0]);
        firstquartile = (x[(n - 1) / 4] + x[n / 4]) / 2.;
        median = (x[(n - 1) / 2] + x[n / 2]) / 2.;
        thirdquartile = (x[n - 1 - (n - 1) / 4] + x[n - 1 - n / 4]) / 2.;
        maximum = double(x[n - 1]);
    }

    ///////////////////////////////////////////////////////////////////////////
    inline std::size_t nedge_min(std::size_t curr, std::size_t t)
    {
        return (std::min)(curr, t);
    }

    inline std::size_t nedge_max(std::size_t curr, std::size_t t)
    {
        return (std::max)(curr, t);
    }

    void print_statistics(std::vector<double> const& kernel2_time,
        std::vector<std::size_t> const& kernel2_nedge)
    {
        // Prep output statistics
        double minimum, mean, stdev;
        double firstquartile, median, thirdquartile, maximum;
        get_statistics(kernel2_time, minimum, mean, stdev,
            firstquartile, median, thirdquartile, maximum);

        std::cout << " min_time:             " << minimum << std::endl;
        std::cout << " firstquartile_time:   " << firstquartile << std::endl;
        std::cout << " median_time:          " << median << std::endl;
        std::cout << " thirdquartile_time:   " << thirdquartile << std::endl;
        std::cout << " max_time:             " << maximum << std::endl;
        std::cout << " mean_time:            " << mean << std::endl;
        std::cout << " stddev_time:          " << stdev << std::endl;

        std::size_t min_edge_num = std::accumulate(
            kernel2_nedge.begin(), kernel2_nedge.end(),
            (std::numeric_limits<std::size_t>::max)(), nedge_min);
        std::size_t max_edge_num = std::accumulate(
            kernel2_nedge.begin(), kernel2_nedge.end(), std::size_t(0), nedge_max);

        if (min_edge_num == max_edge_num) {
            std::cout << " nedge:                " << min_edge_num << std::endl;
        }
        else {
            double n_min, n_mean, n_stdev;
            double n_firstquartile, n_median, n_thirdquartile, n_maximum;
            get_statistics(kernel2_nedge, n_min, n_mean, n_stdev,
                n_firstquartile, n_median, n_thirdquartile, n_maximum);

            std::cout << " min_nedge:            " << n_min << std::endl;
            std::cout << " firstquartile_nedge:  " << n_firstquartile << std::endl;
            std::cout << " median_nedge:         " << n_median << std::endl;
            std::cout << " thirdquartile_nedge:  " << n_thirdquartile << std::endl;
            std::cout << " max_nedge:            " << n_maximum << std::endl;
            std::cout << " mean_nedge:           " << n_mean << std::endl;
            std::cout << " stddev_nedge:         " << n_stdev << std::endl;
        }

        std::vector<double> TEPS;
        TEPS.resize(kernel2_nedge.size());
        for (std::size_t i = 0; i < kernel2_nedge.size(); ++i)
            TEPS[i] = kernel2_nedge[i]/kernel2_time[i];

        double TEPS_min, TEPS_mean, TEPS_stdev;
        double TEPS_firstquartile, TEPS_median, TEPS_thirdquartile, TEPS_maximum;
        get_statistics(TEPS, TEPS_min, TEPS_mean, TEPS_stdev,
            TEPS_firstquartile, TEPS_median, TEPS_thirdquartile, TEPS_maximum);

        // Harmonic standard deviation from:
        // Nilan Norris, The Standard Errors of the Geometric and Harmonic
        // Means and Their Application to Index Numbers, 1940.
        // http://www.jstor.org/stable/2235723
        double N = static_cast<double>(TEPS.size());
        TEPS_stdev /= TEPS_mean * TEPS_mean * std::sqrt(N-1);

        std::cout << " min_TEPS:             " << TEPS_min << std::endl;
        std::cout << " firstquartile_TEPS:   " << TEPS_firstquartile << std::endl;
        std::cout << " median_TEPS:          " << TEPS_median << std::endl;
        std::cout << " thirdquartile_TEPS:   " << TEPS_thirdquartile << std::endl;
        std::cout << " max_TEPS:             " << TEPS_maximum << std::endl;
        std::cout << " harmonic_mean_TEPS:   " << TEPS_mean << std::endl;
        std::cout << " harmonic_stddev_TEPS: " << TEPS_stdev << std::endl;
    }
}


