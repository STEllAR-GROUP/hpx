//  Copyright (c) 2026 Abhishek Bansal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Tests for distributed merge with 4+ localities.
// These exercise complex multi-locality distribution patterns that
// cannot be tested with only 2 localities.
//
// NOTE: container_layout(N, locs) where N is not a multiple of
// locs.size() triggers a partitioned_vector construction bug.
// All tests use partition counts that are multiples of the locality
// count to work around this.

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)

#include <hpx/hpx_main.hpp>
#include <hpx/include/partitioned_vector.hpp>
#include <hpx/include/partitioned_vector_predef.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/modules/algorithms.hpp>
#include <hpx/modules/executors.hpp>
#include <hpx/modules/serialization.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/parallel/segmented_algorithm.hpp>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
struct tagged_value
{
    int key = 0;
    int tag = 0;

    friend bool operator==(tagged_value const& lhs, tagged_value const& rhs)
    {
        return lhs.key == rhs.key && lhs.tag == rhs.tag;
    }

private:
    friend class hpx::serialization::access;

    template <typename Archive>
    void serialize(Archive& ar, unsigned)
    {
        ar & key & tag;
    }
};

std::ostream& operator<<(std::ostream& os, tagged_value const& v)
{
    return os << '{' << v.key << ',' << v.tag << '}';
}

HPX_REGISTER_PARTITIONED_VECTOR_DECLARATION(tagged_value)
HPX_REGISTER_PARTITIONED_VECTOR(tagged_value)

///////////////////////////////////////////////////////////////////////////////
template <typename T>
void assign_vector(hpx::partitioned_vector<T>& v, std::vector<T> const& data)
{
    HPX_TEST_EQ(v.size(), data.size());
    auto it = v.begin();
    for (T const& val : data)
    {
        *it++ = val;
    }
}

template <typename T>
std::vector<T> collect_vector(hpx::partitioned_vector<T> const& v)
{
    std::vector<T> result;
    result.reserve(v.size());
    for (auto it = v.begin(); it != v.end(); ++it)
    {
        result.push_back(*it);
    }
    return result;
}

template <typename T>
void print_sequence(char const* label, std::vector<T> const& values)
{
    std::cerr << label << ": ";
    for (auto const& v : values)
    {
        std::cerr << v << ' ';
    }
    std::cerr << '\n';
}

template <typename T>
void verify_merge(hpx::partitioned_vector<T> const& D,
    std::vector<T> const& expected, char const* test_name)
{
    auto actual = collect_vector(D);
    if (!(actual == expected) && hpx::get_locality_id() == 0)
    {
        std::cerr << "FAIL in " << test_name << ":\n";
        print_sequence("  actual  ", actual);
        print_sequence("  expected", expected);
    }
    HPX_TEST(actual == expected);
}

///////////////////////////////////////////////////////////////////////////////
// Test 1: Each range on a completely disjoint set of localities
// A on {loc0, loc1}, B on {loc2, loc3}, D on all four.
void test_merge_disjoint_ownership()
{
    auto locs = hpx::find_all_localities();
    HPX_TEST(locs.size() >= 4);
    if (locs.size() < 4)
        return;

    std::vector<hpx::id_type> locs_01 = {locs[0], locs[1]};
    std::vector<hpx::id_type> locs_23 = {locs[2], locs[3]};

    std::vector<int> A_data(10);
    for (std::size_t i = 0; i < 10; ++i)
        A_data[i] = static_cast<int>(2 * i);    // 0,2,4,...,18

    std::vector<int> B_data(10);
    for (std::size_t i = 0; i < 10; ++i)
        B_data[i] = static_cast<int>(2 * i + 1);    // 1,3,5,...,19

    hpx::partitioned_vector<int> A(
        A_data.size(), hpx::container_layout(locs_01));
    hpx::partitioned_vector<int> B(
        B_data.size(), hpx::container_layout(locs_23));
    hpx::partitioned_vector<int> D(
        A_data.size() + B_data.size(), hpx::container_layout(locs));

    assign_vector(A, A_data);
    assign_vector(B, B_data);

    std::vector<int> expected;
    expected.reserve(20);
    std::merge(A_data.begin(), A_data.end(), B_data.begin(), B_data.end(),
        std::back_inserter(expected));

    hpx::merge(
        hpx::execution::seq, A.begin(), A.end(), B.begin(), B.end(), D.begin());

    verify_merge(D, expected, "test_merge_disjoint_ownership");
}

///////////////////////////////////////////////////////////////////////////////
// Test 2: Non-contiguous A and D across 4 localities
// A: 8 partitions on 4 localities, each locality owns 2 non-adjacent
// partitions (e.g. loc0 owns partitions {0, 4}).
// D: also 8 partitions to test non-contiguous destination writes.
void test_merge_noncontiguous_4loc()
{
    auto locs = hpx::find_all_localities();
    HPX_TEST(locs.size() >= 4);
    if (locs.size() < 4)
        return;

    // A: 24 elements in 8 partitions (3 each), round-robin
    // loc0: partitions {0,4}, loc1: {1,5}, loc2: {2,6}, loc3: {3,7}
    std::vector<int> A_data(24);
    for (std::size_t i = 0; i < 24; ++i)
        A_data[i] = static_cast<int>(2 * i);    // 0,2,...,46

    // B: 8 elements, 4 partitions (2 each), one per locality
    std::vector<int> B_data{1, 5, 9, 13, 17, 21, 25, 29};

    hpx::partitioned_vector<int> A(
        A_data.size(), hpx::container_layout(8, locs));
    hpx::partitioned_vector<int> B(B_data.size(), hpx::container_layout(locs));
    // D: 32 elements in 8 partitions (4 each)
    hpx::partitioned_vector<int> D(
        A_data.size() + B_data.size(), hpx::container_layout(8, locs));

    assign_vector(A, A_data);
    assign_vector(B, B_data);

    std::vector<int> expected;
    expected.reserve(32);
    std::merge(A_data.begin(), A_data.end(), B_data.begin(), B_data.end(),
        std::back_inserter(expected));

    hpx::merge(
        hpx::execution::seq, A.begin(), A.end(), B.begin(), B.end(), D.begin());

    verify_merge(D, expected, "test_merge_noncontiguous_4loc");
}

///////////////////////////////////////////////////////////////////////////////
// Test 3: A on one locality, B on another, D on two others
// All three ranges have completely different owners.
void test_merge_three_disjoint_owner_sets()
{
    auto locs = hpx::find_all_localities();
    HPX_TEST(locs.size() >= 4);
    if (locs.size() < 4)
        return;

    std::vector<hpx::id_type> loc0 = {locs[0]};
    std::vector<hpx::id_type> loc1 = {locs[1]};
    std::vector<hpx::id_type> locs_23 = {locs[2], locs[3]};

    std::vector<int> const A_data{1, 4, 7, 10, 13, 16};
    std::vector<int> const B_data{2, 5, 8, 11, 14, 17};

    hpx::partitioned_vector<int> A(A_data.size(), hpx::container_layout(loc0));
    hpx::partitioned_vector<int> B(B_data.size(), hpx::container_layout(loc1));
    hpx::partitioned_vector<int> D(
        A_data.size() + B_data.size(), hpx::container_layout(locs_23));

    assign_vector(A, A_data);
    assign_vector(B, B_data);

    std::vector<int> expected;
    expected.reserve(12);
    std::merge(A_data.begin(), A_data.end(), B_data.begin(), B_data.end(),
        std::back_inserter(expected));

    hpx::merge(
        hpx::execution::seq, A.begin(), A.end(), B.begin(), B.end(), D.begin());

    verify_merge(D, expected, "test_merge_three_disjoint_owner_sets");
}

///////////////////////////////////////////////////////////////////////////////
// Test 4: Many small partitions across 4 localities
// 8 partitions per range = 2 per locality. Tests the algorithm with
// many segments and many intervals in the all_gather/all_to_all.
void test_merge_many_partitions_4loc()
{
    auto locs = hpx::find_all_localities();
    HPX_TEST(locs.size() >= 4);
    if (locs.size() < 4)
        return;

    std::vector<int> A_data(24);
    for (std::size_t i = 0; i < 24; ++i)
        A_data[i] = static_cast<int>(2 * i);

    std::vector<int> B_data(24);
    for (std::size_t i = 0; i < 24; ++i)
        B_data[i] = static_cast<int>(2 * i + 1);

    hpx::partitioned_vector<int> A(
        A_data.size(), hpx::container_layout(8, locs));
    hpx::partitioned_vector<int> B(
        B_data.size(), hpx::container_layout(8, locs));
    hpx::partitioned_vector<int> D(
        A_data.size() + B_data.size(), hpx::container_layout(8, locs));

    assign_vector(A, A_data);
    assign_vector(B, B_data);

    std::vector<int> expected;
    expected.reserve(48);
    std::merge(A_data.begin(), A_data.end(), B_data.begin(), B_data.end(),
        std::back_inserter(expected));

    hpx::merge(
        hpx::execution::seq, A.begin(), A.end(), B.begin(), B.end(), D.begin());

    verify_merge(D, expected, "test_merge_many_partitions_4loc");
}

///////////////////////////////////////////////////////////////////////////////
// Test 5: Stability with 4 localities and tagged values
// Heavy duplicate keys spread across multiple localities.
void test_merge_stability_4loc()
{
    auto locs = hpx::find_all_localities();
    HPX_TEST(locs.size() >= 4);
    if (locs.size() < 4)
        return;

    using vector_type = hpx::partitioned_vector<tagged_value>;

    // Both A and B must be globally sorted by key across all partitions.
    // A: 16 elements, 4 partitions (4 each), one per locality
    std::vector<tagged_value> A_data{
        {1, 0}, {2, 0}, {3, 0}, {4, 0},       // loc0
        {5, 0}, {6, 0}, {7, 0}, {8, 0},       // loc1
        {9, 0}, {10, 0}, {11, 0}, {12, 0},    // loc2
        {13, 0}, {14, 0}, {15, 0}, {16, 0}    // loc3
    };

    // B: 8 elements, 4 partitions (2 each): duplicate keys with A
    std::vector<tagged_value> B_data{
        {1, 1}, {2, 1},     // loc0
        {5, 1}, {6, 1},     // loc1
        {9, 1}, {10, 1},    // loc2
        {13, 1}, {14, 1}    // loc3
    };

    vector_type A(A_data.size(), hpx::container_layout(locs));
    vector_type B(B_data.size(), hpx::container_layout(locs));
    vector_type D(A_data.size() + B_data.size(), hpx::container_layout(locs));

    assign_vector(A, A_data);
    assign_vector(B, B_data);

    auto comp = [](tagged_value const& lhs, tagged_value const& rhs) {
        return lhs.key < rhs.key;
    };

    std::vector<tagged_value> expected;
    expected.reserve(24);
    std::merge(A_data.begin(), A_data.end(), B_data.begin(), B_data.end(),
        std::back_inserter(expected), comp);

    hpx::merge(hpx::execution::seq, A.begin(), A.end(), B.begin(), B.end(),
        D.begin(), comp);

    verify_merge(D, expected, "test_merge_stability_4loc");
}

///////////////////////////////////////////////////////////////////////////////
// Test 6: Skewed: one locality has all the data, others have nothing
// A on loc0 only, B on loc0 only, D spread across all 4.
void test_merge_single_source_multi_dest()
{
    auto locs = hpx::find_all_localities();
    HPX_TEST(locs.size() >= 4);
    if (locs.size() < 4)
        return;

    std::vector<hpx::id_type> loc0 = {locs[0]};

    std::vector<int> const A_data{1, 3, 5, 7, 9, 11, 13, 15};
    std::vector<int> const B_data{2, 4, 6, 8, 10, 12, 14, 16};

    hpx::partitioned_vector<int> A(A_data.size(), hpx::container_layout(loc0));
    hpx::partitioned_vector<int> B(B_data.size(), hpx::container_layout(loc0));
    hpx::partitioned_vector<int> D(
        A_data.size() + B_data.size(), hpx::container_layout(locs));

    assign_vector(A, A_data);
    assign_vector(B, B_data);

    std::vector<int> expected;
    expected.reserve(16);
    std::merge(A_data.begin(), A_data.end(), B_data.begin(), B_data.end(),
        std::back_inserter(expected));

    hpx::merge(
        hpx::execution::seq, A.begin(), A.end(), B.begin(), B.end(), D.begin());

    verify_merge(D, expected, "test_merge_single_source_multi_dest");
}

///////////////////////////////////////////////////////////////////////////////
// Test 7: Reverse: D on one locality, A and B spread across 4.
// loc0 is the sole dest owner, must receive payload from all others.
void test_merge_multi_source_single_dest()
{
    auto locs = hpx::find_all_localities();
    HPX_TEST(locs.size() >= 4);
    if (locs.size() < 4)
        return;

    std::vector<hpx::id_type> loc0 = {locs[0]};

    std::vector<int> A_data(16);
    for (std::size_t i = 0; i < 16; ++i)
        A_data[i] = static_cast<int>(2 * i);

    std::vector<int> B_data(16);
    for (std::size_t i = 0; i < 16; ++i)
        B_data[i] = static_cast<int>(2 * i + 1);

    hpx::partitioned_vector<int> A(A_data.size(), hpx::container_layout(locs));
    hpx::partitioned_vector<int> B(B_data.size(), hpx::container_layout(locs));
    hpx::partitioned_vector<int> D(
        A_data.size() + B_data.size(), hpx::container_layout(loc0));

    assign_vector(A, A_data);
    assign_vector(B, B_data);

    std::vector<int> expected;
    expected.reserve(32);
    std::merge(A_data.begin(), A_data.end(), B_data.begin(), B_data.end(),
        std::back_inserter(expected));

    hpx::merge(
        hpx::execution::seq, A.begin(), A.end(), B.begin(), B.end(), D.begin());

    verify_merge(D, expected, "test_merge_multi_source_single_dest");
}

///////////////////////////////////////////////////////////////////////////////
// Test 8: par(task) with 4 localities: async path
void test_merge_async_4loc()
{
    auto locs = hpx::find_all_localities();
    HPX_TEST(locs.size() >= 4);
    if (locs.size() < 4)
        return;

    std::vector<int> A_data(20);
    std::vector<int> B_data(20);
    for (std::size_t i = 0; i < 20; ++i)
    {
        A_data[i] = static_cast<int>(2 * i);
        B_data[i] = static_cast<int>(2 * i + 1);
    }

    hpx::partitioned_vector<int> A(A_data.size(), hpx::container_layout(locs));
    hpx::partitioned_vector<int> B(B_data.size(), hpx::container_layout(locs));
    hpx::partitioned_vector<int> D(
        A_data.size() + B_data.size(), hpx::container_layout(locs));

    assign_vector(A, A_data);
    assign_vector(B, B_data);

    std::vector<int> expected;
    expected.reserve(40);
    std::merge(A_data.begin(), A_data.end(), B_data.begin(), B_data.end(),
        std::back_inserter(expected));

    auto result = hpx::merge(hpx::execution::par(hpx::execution::task),
        A.begin(), A.end(), B.begin(), B.end(), D.begin());

    HPX_TEST(result.get() == D.end());
    verify_merge(D, expected, "test_merge_async_4loc");
}

///////////////////////////////////////////////////////////////////////////////
// Test 9: Larger stress test: 200 elements, many partitions, 4 locs
void test_merge_stress_4loc()
{
    auto locs = hpx::find_all_localities();
    HPX_TEST(locs.size() >= 4);
    if (locs.size() < 4)
        return;

    std::size_t const N_A = 120;
    std::size_t const N_B = 80;

    std::vector<int> A_data(N_A);
    std::vector<int> B_data(N_B);

    for (std::size_t i = 0; i < N_A; ++i)
        A_data[i] = static_cast<int>(i * 3);

    for (std::size_t i = 0; i < N_B; ++i)
        B_data[i] = static_cast<int>(i * 3 + 1);

    // 8 partitions round-robin across 4 locs = 2 per loc per range
    hpx::partitioned_vector<int> A(N_A, hpx::container_layout(8, locs));
    hpx::partitioned_vector<int> B(N_B, hpx::container_layout(8, locs));
    hpx::partitioned_vector<int> D(N_A + N_B, hpx::container_layout(8, locs));

    assign_vector(A, A_data);
    assign_vector(B, B_data);

    std::vector<int> expected;
    expected.reserve(N_A + N_B);
    std::merge(A_data.begin(), A_data.end(), B_data.begin(), B_data.end(),
        std::back_inserter(expected));

    hpx::merge(
        hpx::execution::seq, A.begin(), A.end(), B.begin(), B.end(), D.begin());

    verify_merge(D, expected, "test_merge_stress_4loc");
}

///////////////////////////////////////////////////////////////////////////////
// Test 10: Skewed sizes with non-contiguous partitions
// A: large (32 elems, 8 partitions), B: small (4 elems, 4 partitions)
// Tests co-rank when one input dominates.
void test_merge_skewed_noncontiguous()
{
    auto locs = hpx::find_all_localities();
    HPX_TEST(locs.size() >= 4);
    if (locs.size() < 4)
        return;

    std::vector<int> A_data(32);
    for (std::size_t i = 0; i < 32; ++i)
        A_data[i] = static_cast<int>(i);    // 0..31

    std::vector<int> B_data{5, 15, 25, 35};

    hpx::partitioned_vector<int> A(
        A_data.size(), hpx::container_layout(8, locs));
    hpx::partitioned_vector<int> B(B_data.size(), hpx::container_layout(locs));
    hpx::partitioned_vector<int> D(
        A_data.size() + B_data.size(), hpx::container_layout(locs));

    assign_vector(A, A_data);
    assign_vector(B, B_data);

    std::vector<int> expected;
    expected.reserve(36);
    std::merge(A_data.begin(), A_data.end(), B_data.begin(), B_data.end(),
        std::back_inserter(expected));

    hpx::merge(
        hpx::execution::seq, A.begin(), A.end(), B.begin(), B.end(), D.begin());

    verify_merge(D, expected, "test_merge_skewed_noncontiguous");
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    test_merge_disjoint_ownership();
    test_merge_noncontiguous_4loc();
    test_merge_three_disjoint_owner_sets();
    test_merge_many_partitions_4loc();
    test_merge_stability_4loc();
    test_merge_single_source_multi_dest();
    test_merge_multi_source_single_dest();
    test_merge_async_4loc();
    test_merge_stress_4loc();
    test_merge_skewed_noncontiguous();

    return hpx::util::report_errors();
}

#endif
