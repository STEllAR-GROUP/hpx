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

///////////////////////////////////////////////////////////////////////////////
// this routine mirrors the matlab validation routine

inline std::size_t
max_node(std::size_t n1, std::pair<std::size_t, std::size_t> const& p)
{
    return (std::max)((std::max)(n1, p.first), p.second);
}

int validate(std::size_t searchkey,
    std::vector<std::size_t> const& parents,
    std::vector<std::pair<std::size_t, std::size_t> > const& edgelist,
    std::size_t& num_nodes)
{
    // find the largest edge number
    // octave:  N = max (max (ij));
    std::size_t N = std::accumulate(edgelist.begin(), edgelist.end(),
        std::size_t(0), max_node) + 1;

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

    // octave: P = parent (slice);
    std::vector<std::size_t> P(slice.size());
    for (std::size_t i=0;i<slice.size();i++)
        P[i] = parents[slice[i]];

    // Define a mask
    // octave:  mask = P != search_key;
    boost::dynamic_bitset<> mask(slice.size());
    for (std::size_t i = 0; i < slice.size(); ++i) {
        if (P[i] != searchkey) {
            mask[i] = true;
        }
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
        for (std::size_t i = 0; i < P.size(); ++i)
            P[i] = parents[P[i]];

        for (std::size_t i = 0; i < P.size(); ++i)
            mask[i] = P[i] != searchkey;

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

///////////////////////////////////////////////////////////////////////////////
struct stddev
{
    stddev(double mean) : mean_(mean) {}

    double operator()(double curr, double x) const
    {
        return curr + (x - mean_) * (x  - mean_);
    }

    double mean_;
};

void get_statistics(std::vector<double> x,
    double &minimum, double &mean, double &deviation, double &firstquartile,
    double &median, double &thirdquartile, double &maximum)
{
    // Compute mean
    std::size_t n = x.size();
    mean = std::accumulate(x.begin(), x.end(), 0.0) / n;

    // Compute std deviation
    deviation = std::accumulate(x.begin(), x.end(), 0.0, stddev(mean)) / std::sqrt(n - 1.0);

    // Sort x
    std::sort(x.begin(), x.end());

    minimum = x[0];
    firstquartile = (x[(n - 1) / 4] + x[n / 4]) / 2;
    median = (x[(n - 1) / 2] + x[n / 2]) / 2;
    thirdquartile = (x[n - 1 - (n - 1) / 4] + x[n - 1 - n / 4]) / 2;
    maximum = x[n - 1];
}

