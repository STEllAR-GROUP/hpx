//  Copyright (c) 2026 Abhishek Bansal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

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
#include <numeric>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
// A tagged value type for testing merge stability (A-before-B on equal keys)
struct tagged_value
{
    int key = 0;
    int tag = 0;    // 0 = from A, 1 = from B

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
// Helpers
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

// Verify merge result against std::merge reference, with diagnostics
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
// Test 1: Stability — A elements come before B on equal keys
template <typename ExPolicy>
void test_merge_stability(ExPolicy&& policy)
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();

    using vector_type = hpx::partitioned_vector<tagged_value>;

    std::vector<tagged_value> const A_data{
        {1, 0}, {2, 0}, {2, 0}, {3, 0}, {5, 0}, {5, 0}, {8, 0}, {13, 0}};
    std::vector<tagged_value> const B_data{
        {1, 1}, {2, 1}, {2, 1}, {4, 1}, {5, 1}, {9, 1}, {13, 1}};

    vector_type A(A_data.size(), hpx::container_layout(localities));
    vector_type B(B_data.size(), hpx::container_layout(localities));
    vector_type D(
        A_data.size() + B_data.size(), hpx::container_layout(localities));

    assign_vector(A, A_data);
    assign_vector(B, B_data);

    auto comp = [](tagged_value const& lhs, tagged_value const& rhs) {
        return lhs.key < rhs.key;
    };

    std::vector<tagged_value> expected;
    expected.reserve(D.size());
    std::merge(A_data.begin(), A_data.end(), B_data.begin(), B_data.end(),
        std::back_inserter(expected), comp);

    auto result = hpx::merge(HPX_FORWARD(ExPolicy, policy), A.begin(), A.end(),
        B.begin(), B.end(), D.begin(), comp);

    if constexpr (hpx::is_async_execution_policy_v<std::decay_t<ExPolicy>>)
    {
        HPX_TEST(result.get() == D.end());
    }
    else
    {
        HPX_TEST(result == D.end());
    }

    verify_merge(D, expected, "test_merge_stability");
}

///////////////////////////////////////////////////////////////////////////////
// Test 2: No-policy overload
void test_merge_no_policy()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();

    using vector_type = hpx::partitioned_vector<tagged_value>;

    std::vector<tagged_value> const A_data{
        {0, 0}, {1, 0}, {1, 0}, {4, 0}, {4, 0}, {10, 0}};
    std::vector<tagged_value> const B_data{
        {1, 1}, {1, 1}, {2, 1}, {4, 1}, {9, 1}};

    vector_type A(A_data.size(), hpx::container_layout(localities));
    vector_type B(B_data.size(), hpx::container_layout(localities));
    vector_type D(
        A_data.size() + B_data.size(), hpx::container_layout(localities));

    assign_vector(A, A_data);
    assign_vector(B, B_data);

    auto comp = [](tagged_value const& lhs, tagged_value const& rhs) {
        return lhs.key < rhs.key;
    };

    std::vector<tagged_value> expected;
    expected.reserve(D.size());
    std::merge(A_data.begin(), A_data.end(), B_data.begin(), B_data.end(),
        std::back_inserter(expected), comp);

    HPX_TEST(hpx::merge(A.begin(), A.end(), B.begin(), B.end(), D.begin(),
                 comp) == D.end());
    verify_merge(D, expected, "test_merge_no_policy");
}

///////////////////////////////////////////////////////////////////////////////
// Test 3: Mixed types (int + long long -> long long)
template <typename ExPolicy>
void test_merge_mixed_types(ExPolicy&& policy)
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();

    hpx::partitioned_vector<int> A(6, hpx::container_layout(localities));
    hpx::partitioned_vector<long long> B(5, hpx::container_layout(localities));
    hpx::partitioned_vector<long long> D(11, hpx::container_layout(localities));

    std::vector<int> const A_data{1, 2, 2, 5, 8, 13};
    std::vector<long long> const B_data{0, 2, 3, 8, 21};

    assign_vector(A, A_data);
    assign_vector(B, B_data);

    auto comp = [](auto const& lhs, auto const& rhs) { return lhs < rhs; };

    std::vector<long long> expected;
    expected.reserve(D.size());
    std::merge(A_data.begin(), A_data.end(), B_data.begin(), B_data.end(),
        std::back_inserter(expected), comp);

    auto result = hpx::merge(HPX_FORWARD(ExPolicy, policy), A.begin(), A.end(),
        B.begin(), B.end(), D.begin(), comp);

    if constexpr (hpx::is_async_execution_policy_v<std::decay_t<ExPolicy>>)
    {
        HPX_TEST(result.get() == D.end());
    }
    else
    {
        HPX_TEST(result == D.end());
    }

    verify_merge(D, expected, "test_merge_mixed_types");
}

///////////////////////////////////////////////////////////////////////////////
// Test 4: Empty inputs
template <typename ExPolicy>
void test_merge_empty_inputs(ExPolicy&& policy)
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    auto layout = hpx::container_layout(localities);

    // Empty A, non-empty B
    {
        hpx::partitioned_vector<int> A_empty(0, layout);
        hpx::partitioned_vector<int> B(5, layout);
        hpx::partitioned_vector<int> D(5, layout);

        std::vector<int> const B_data{1, 2, 3, 5, 8};
        assign_vector(B, B_data);

        auto result = hpx::merge(HPX_FORWARD(ExPolicy, policy), A_empty.begin(),
            A_empty.end(), B.begin(), B.end(), D.begin());

        if constexpr (hpx::is_async_execution_policy_v<std::decay_t<ExPolicy>>)
        {
            HPX_TEST(result.get() == D.end());
        }
        else
        {
            HPX_TEST(result == D.end());
        }

        verify_merge(D, B_data, "test_merge_empty_A");
    }

    // Non-empty A, empty B
    {
        hpx::partitioned_vector<int> A(4, layout);
        hpx::partitioned_vector<int> B_empty(0, layout);
        hpx::partitioned_vector<int> D(4, layout);

        std::vector<int> const A_data{0, 1, 1, 2};
        assign_vector(A, A_data);

        auto result = hpx::merge(HPX_FORWARD(ExPolicy, policy), A.begin(),
            A.end(), B_empty.begin(), B_empty.end(), D.begin());

        if constexpr (hpx::is_async_execution_policy_v<std::decay_t<ExPolicy>>)
        {
            HPX_TEST(result.get() == D.end());
        }
        else
        {
            HPX_TEST(result == D.end());
        }

        verify_merge(D, A_data, "test_merge_empty_B");
    }
}

///////////////////////////////////////////////////////////////////////////////
// Test 5: Both inputs empty
void test_merge_both_empty()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    auto layout = hpx::container_layout(localities);

    hpx::partitioned_vector<int> A(0, layout);
    hpx::partitioned_vector<int> B(0, layout);
    hpx::partitioned_vector<int> D(0, layout);

    auto result = hpx::merge(
        hpx::execution::seq, A.begin(), A.end(), B.begin(), B.end(), D.begin());

    HPX_TEST(result == D.end());
    HPX_TEST(D.size() == 0);
}

///////////////////////////////////////////////////////////////////////////////
// Test 6: Single element in each input
void test_merge_single_elements()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    auto layout = hpx::container_layout(localities);

    // A=[3], B=[1] -> D=[1,3]
    {
        hpx::partitioned_vector<int> A(1, layout);
        hpx::partitioned_vector<int> B(1, layout);
        hpx::partitioned_vector<int> D(2, layout);

        assign_vector(A, std::vector<int>{3});
        assign_vector(B, std::vector<int>{1});

        hpx::merge(hpx::execution::seq, A.begin(), A.end(), B.begin(), B.end(),
            D.begin());

        verify_merge(D, std::vector<int>{1, 3}, "test_merge_single_3_1");
    }

    // A=[1], B=[1] -> D=[1,1] (stability: A first)
    {
        using vector_type = hpx::partitioned_vector<tagged_value>;

        vector_type A(1, layout);
        vector_type B(1, layout);
        vector_type D(2, layout);

        assign_vector(A, std::vector<tagged_value>{{5, 0}});
        assign_vector(B, std::vector<tagged_value>{{5, 1}});

        auto comp = [](tagged_value const& lhs, tagged_value const& rhs) {
            return lhs.key < rhs.key;
        };

        hpx::merge(hpx::execution::seq, A.begin(), A.end(), B.begin(), B.end(),
            D.begin(), comp);

        // Stability: A's element ({5,0}) must come before B's ({5,1})
        std::vector<tagged_value> expected{{5, 0}, {5, 1}};
        verify_merge(D, expected, "test_merge_single_equal_stability");
    }
}

///////////////////////////////////////////////////////////////////////////////
// Test 7: All duplicate keys — heavy stability test
void test_merge_all_duplicates()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    auto layout = hpx::container_layout(localities);

    using vector_type = hpx::partitioned_vector<tagged_value>;

    // 6 elements in A all with key=7, 4 elements in B all with key=7
    std::vector<tagged_value> A_data(6, {7, 0});
    std::vector<tagged_value> B_data(4, {7, 1});

    vector_type A(A_data.size(), layout);
    vector_type B(B_data.size(), layout);
    vector_type D(A_data.size() + B_data.size(), layout);

    assign_vector(A, A_data);
    assign_vector(B, B_data);

    auto comp = [](tagged_value const& lhs, tagged_value const& rhs) {
        return lhs.key < rhs.key;
    };

    // Stable merge: all 6 A elements (tag=0) then all 4 B elements (tag=1)
    std::vector<tagged_value> expected;
    expected.reserve(10);
    std::merge(A_data.begin(), A_data.end(), B_data.begin(), B_data.end(),
        std::back_inserter(expected), comp);

    hpx::merge(hpx::execution::seq, A.begin(), A.end(), B.begin(), B.end(),
        D.begin(), comp);

    verify_merge(D, expected, "test_merge_all_duplicates");
}

///////////////////////////////////////////////////////////////////////////////
// Test 8: Non-overlapping ranges (A all less than B)
void test_merge_nonoverlapping_A_less()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    auto layout = hpx::container_layout(localities);

    std::vector<int> const A_data{1, 2, 3, 4};
    std::vector<int> const B_data{10, 11, 12, 13};

    hpx::partitioned_vector<int> A(A_data.size(), layout);
    hpx::partitioned_vector<int> B(B_data.size(), layout);
    hpx::partitioned_vector<int> D(A_data.size() + B_data.size(), layout);

    assign_vector(A, A_data);
    assign_vector(B, B_data);

    hpx::merge(
        hpx::execution::seq, A.begin(), A.end(), B.begin(), B.end(), D.begin());

    std::vector<int> expected{1, 2, 3, 4, 10, 11, 12, 13};
    verify_merge(D, expected, "test_merge_nonoverlapping_A_less");
}

///////////////////////////////////////////////////////////////////////////////
// Test 9: Non-overlapping ranges (A all greater than B)
void test_merge_nonoverlapping_A_greater()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    auto layout = hpx::container_layout(localities);

    std::vector<int> const A_data{10, 11, 12, 13};
    std::vector<int> const B_data{1, 2, 3, 4};

    hpx::partitioned_vector<int> A(A_data.size(), layout);
    hpx::partitioned_vector<int> B(B_data.size(), layout);
    hpx::partitioned_vector<int> D(A_data.size() + B_data.size(), layout);

    assign_vector(A, A_data);
    assign_vector(B, B_data);

    hpx::merge(
        hpx::execution::seq, A.begin(), A.end(), B.begin(), B.end(), D.begin());

    std::vector<int> expected{1, 2, 3, 4, 10, 11, 12, 13};
    verify_merge(D, expected, "test_merge_nonoverlapping_A_greater");
}

///////////////////////////////////////////////////////////////////////////////
// Test 10: Highly skewed sizes (A much larger than B)
void test_merge_skewed_sizes()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    auto layout = hpx::container_layout(localities);

    // A has 20 elements, B has 2
    std::vector<int> A_data(20);
    std::iota(A_data.begin(), A_data.end(), 0);    // 0..19

    std::vector<int> const B_data{5, 15};

    hpx::partitioned_vector<int> A(A_data.size(), layout);
    hpx::partitioned_vector<int> B(B_data.size(), layout);
    hpx::partitioned_vector<int> D(A_data.size() + B_data.size(), layout);

    assign_vector(A, A_data);
    assign_vector(B, B_data);

    std::vector<int> expected;
    expected.reserve(D.size());
    std::merge(A_data.begin(), A_data.end(), B_data.begin(), B_data.end(),
        std::back_inserter(expected));

    hpx::merge(
        hpx::execution::seq, A.begin(), A.end(), B.begin(), B.end(), D.begin());

    verify_merge(D, expected, "test_merge_skewed_sizes");
}

///////////////////////////////////////////////////////////////////////////////
// Test 11: Default comparator (no custom comp — uses hpx::parallel::detail::less)
void test_merge_default_comp()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    auto layout = hpx::container_layout(localities);

    std::vector<int> const A_data{1, 3, 5, 7, 9};
    std::vector<int> const B_data{2, 4, 6, 8, 10};

    hpx::partitioned_vector<int> A(A_data.size(), layout);
    hpx::partitioned_vector<int> B(B_data.size(), layout);
    hpx::partitioned_vector<int> D(A_data.size() + B_data.size(), layout);

    assign_vector(A, A_data);
    assign_vector(B, B_data);

    // Call without explicit comparator — uses default less
    hpx::merge(
        hpx::execution::seq, A.begin(), A.end(), B.begin(), B.end(), D.begin());

    std::vector<int> expected{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    verify_merge(D, expected, "test_merge_default_comp");
}

///////////////////////////////////////////////////////////////////////////////
// Test 12: Non-contiguous A partitions on one locality
// With 3 partitions on 2 localities, round-robin gives:
//   partition 0 -> loc0, partition 1 -> loc1, partition 2 -> loc0
// So loc0 owns partitions 0 and 2 (non-contiguous in global order).
// This exercises the slice-based design where one locality has
// discontiguous pieces of a range.
void test_merge_noncontiguous_A()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    if (localities.size() < 2)
        return;    // need 2+ localities

    using vector_type = hpx::partitioned_vector<tagged_value>;

    // A: 12 elements in 3 partitions (4 each) -> [loc0, loc1, loc0]
    // loc0 owns A[0..4) and A[8..12) — non-contiguous
    std::vector<tagged_value> const A_data{
        {1, 0}, {3, 0}, {5, 0}, {7, 0},      // partition 0 on loc0
        {2, 0}, {4, 0}, {6, 0}, {8, 0},      // partition 1 on loc1
        {9, 0}, {10, 0}, {11, 0}, {12, 0}    // partition 2 on loc0
    };

    // B: 6 elements in 2 partitions (3 each) -> [loc0, loc1]
    std::vector<tagged_value> const B_data{
        {2, 1}, {6, 1}, {10, 1},    // partition 0 on loc0
        {4, 1}, {8, 1}, {12, 1}     // partition 1 on loc1
    };

    vector_type A(A_data.size(), hpx::container_layout(3, localities));
    vector_type B(B_data.size(), hpx::container_layout(localities));
    vector_type D(
        A_data.size() + B_data.size(), hpx::container_layout(localities));

    assign_vector(A, A_data);
    assign_vector(B, B_data);

    auto comp = [](tagged_value const& lhs, tagged_value const& rhs) {
        return lhs.key < rhs.key;
    };

    std::vector<tagged_value> expected;
    expected.reserve(D.size());
    std::merge(A_data.begin(), A_data.end(), B_data.begin(), B_data.end(),
        std::back_inserter(expected), comp);

    hpx::merge(hpx::execution::seq, A.begin(), A.end(), B.begin(), B.end(),
        D.begin(), comp);

    verify_merge(D, expected, "test_merge_noncontiguous_A");
}

///////////////////////////////////////////////////////////////////////////////
// Test 13: Non-contiguous D partitions on one locality
// D has 3 partitions on 2 localities: [loc0, loc1, loc0]
// So loc0 writes to dest[0..N/3) and dest[2*N/3..N) — non-contiguous.
void test_merge_noncontiguous_D()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    if (localities.size() < 2)
        return;

    // A: 6 elements in 2 partitions -> [loc0, loc1]
    std::vector<int> const A_data{1, 3, 5, 7, 9, 11};

    // B: 6 elements in 2 partitions -> [loc0, loc1]
    std::vector<int> const B_data{2, 4, 6, 8, 10, 12};

    // D: 12 elements in 3 partitions (4 each) -> [loc0, loc1, loc0]
    // loc0 owns D[0..4) and D[8..12) — non-contiguous destination
    hpx::partitioned_vector<int> A(
        A_data.size(), hpx::container_layout(localities));
    hpx::partitioned_vector<int> B(
        B_data.size(), hpx::container_layout(localities));
    hpx::partitioned_vector<int> D(
        A_data.size() + B_data.size(), hpx::container_layout(3, localities));

    assign_vector(A, A_data);
    assign_vector(B, B_data);

    std::vector<int> expected;
    expected.reserve(D.size());
    std::merge(A_data.begin(), A_data.end(), B_data.begin(), B_data.end(),
        std::back_inserter(expected));

    hpx::merge(
        hpx::execution::seq, A.begin(), A.end(), B.begin(), B.end(), D.begin());

    verify_merge(D, expected, "test_merge_noncontiguous_D");
}

///////////////////////////////////////////////////////////////////////////////
// Test 14: Different ownership — A only on loc0, B only on loc1, D on both
// This forces remote co-rank probes: loc0 must probe B data on loc1,
// and loc1 must probe A data on loc0.
void test_merge_different_ownership()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    if (localities.size() < 2)
        return;

    std::vector<hpx::id_type> loc0_only = {localities[0]};
    std::vector<hpx::id_type> loc1_only = {localities[1]};

    // A: 8 elements, single partition on loc0
    std::vector<int> const A_data{1, 3, 5, 7, 9, 11, 13, 15};

    // B: 6 elements, single partition on loc1
    std::vector<int> const B_data{2, 4, 6, 8, 10, 12};

    hpx::partitioned_vector<int> A(
        A_data.size(), hpx::container_layout(loc0_only));
    hpx::partitioned_vector<int> B(
        B_data.size(), hpx::container_layout(loc1_only));
    hpx::partitioned_vector<int> D(
        A_data.size() + B_data.size(), hpx::container_layout(localities));

    assign_vector(A, A_data);
    assign_vector(B, B_data);

    std::vector<int> expected;
    expected.reserve(D.size());
    std::merge(A_data.begin(), A_data.end(), B_data.begin(), B_data.end(),
        std::back_inserter(expected));

    hpx::merge(
        hpx::execution::seq, A.begin(), A.end(), B.begin(), B.end(), D.begin());

    verify_merge(D, expected, "test_merge_different_ownership");
}

///////////////////////////////////////////////////////////////////////////////
// Test 15: All three ranges on completely different localities
// A on loc0, B on loc1, D on loc0 (but D's owner is also an A owner,
// so it must fetch B data remotely for co-rank and for payload).
void test_merge_A_and_D_on_loc0_B_on_loc1()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    if (localities.size() < 2)
        return;

    std::vector<hpx::id_type> loc0_only = {localities[0]};
    std::vector<hpx::id_type> loc1_only = {localities[1]};

    std::vector<int> const A_data{1, 4, 7, 10};
    std::vector<int> const B_data{2, 5, 8, 11};

    hpx::partitioned_vector<int> A(
        A_data.size(), hpx::container_layout(loc0_only));
    hpx::partitioned_vector<int> B(
        B_data.size(), hpx::container_layout(loc1_only));
    // D entirely on loc0 — loc0 must receive all B payload from loc1
    hpx::partitioned_vector<int> D(
        A_data.size() + B_data.size(), hpx::container_layout(loc0_only));

    assign_vector(A, A_data);
    assign_vector(B, B_data);

    std::vector<int> expected;
    expected.reserve(D.size());
    std::merge(A_data.begin(), A_data.end(), B_data.begin(), B_data.end(),
        std::back_inserter(expected));

    hpx::merge(
        hpx::execution::seq, A.begin(), A.end(), B.begin(), B.end(), D.begin());

    verify_merge(D, expected, "test_merge_A_D_loc0_B_loc1");
}

///////////////////////////////////////////////////////////////////////////////
// Test 16: Multiple non-contiguous partitions for BOTH A and D
// A: 4 partitions on 2 locs -> [loc0, loc1, loc0, loc1]
// D: 3 partitions on 2 locs -> [loc0, loc1, loc0]
// Both A and D have discontiguous slices on loc0.
void test_merge_noncontiguous_A_and_D()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    if (localities.size() < 2)
        return;

    using vector_type = hpx::partitioned_vector<tagged_value>;

    // A: 8 elements in 4 partitions (2 each)
    // [loc0, loc1, loc0, loc1]: loc0 has partitions {0,2}, loc1 has {1,3}
    std::vector<tagged_value> const A_data{
        {1, 0}, {5, 0},     // partition 0 on loc0
        {2, 0}, {6, 0},     // partition 1 on loc1
        {9, 0}, {13, 0},    // partition 2 on loc0
        {10, 0}, {14, 0}    // partition 3 on loc1
    };

    // B: 4 elements in 2 partitions (2 each) -> [loc0, loc1]
    std::vector<tagged_value> const B_data{
        {3, 1}, {7, 1},     // partition 0 on loc0
        {11, 1}, {15, 1}    // partition 1 on loc1
    };

    // D: 12 elements in 3 partitions (4 each) -> [loc0, loc1, loc0]
    // loc0 has D partitions {0,2} — non-contiguous destination
    vector_type A(A_data.size(), hpx::container_layout(4, localities));
    vector_type B(B_data.size(), hpx::container_layout(localities));
    vector_type D(
        A_data.size() + B_data.size(), hpx::container_layout(3, localities));

    assign_vector(A, A_data);
    assign_vector(B, B_data);

    auto comp = [](tagged_value const& lhs, tagged_value const& rhs) {
        return lhs.key < rhs.key;
    };

    std::vector<tagged_value> expected;
    expected.reserve(D.size());
    std::merge(A_data.begin(), A_data.end(), B_data.begin(), B_data.end(),
        std::back_inserter(expected), comp);

    hpx::merge(hpx::execution::seq, A.begin(), A.end(), B.begin(), B.end(),
        D.begin(), comp);

    verify_merge(D, expected, "test_merge_noncontiguous_A_and_D");
}

///////////////////////////////////////////////////////////////////////////////
// Test 17: Participant owns no slices of one or more ranges
// A on loc0, B on loc0, D on loc1
// loc1 owns no A and no B slices, but owns all D slices.
// It must receive all payload from loc0.
// loc0 owns all input but no D slices — it sends everything, writes nothing.
void test_merge_participant_no_input_slices()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    if (localities.size() < 2)
        return;

    std::vector<hpx::id_type> loc0_only = {localities[0]};
    std::vector<hpx::id_type> loc1_only = {localities[1]};

    std::vector<int> const A_data{1, 3, 5, 7};
    std::vector<int> const B_data{2, 4, 6, 8};

    hpx::partitioned_vector<int> A(
        A_data.size(), hpx::container_layout(loc0_only));
    hpx::partitioned_vector<int> B(
        B_data.size(), hpx::container_layout(loc0_only));
    // D entirely on loc1
    hpx::partitioned_vector<int> D(
        A_data.size() + B_data.size(), hpx::container_layout(loc1_only));

    assign_vector(A, A_data);
    assign_vector(B, B_data);

    std::vector<int> expected;
    expected.reserve(D.size());
    std::merge(A_data.begin(), A_data.end(), B_data.begin(), B_data.end(),
        std::back_inserter(expected));

    hpx::merge(
        hpx::execution::seq, A.begin(), A.end(), B.begin(), B.end(), D.begin());

    verify_merge(D, expected, "test_merge_participant_no_input_slices");
}

///////////////////////////////////////////////////////////////////////////////
// Test 18: Interleaved data with many duplicates at boundary
void test_merge_boundary_duplicates()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    auto layout = hpx::container_layout(localities);

    using vector_type = hpx::partitioned_vector<tagged_value>;

    // Keys cluster at value 5 — tests co-rank probe behavior
    // around dense duplicate regions
    std::vector<tagged_value> const A_data{
        {1, 0}, {5, 0}, {5, 0}, {5, 0}, {5, 0}, {10, 0}};
    std::vector<tagged_value> const B_data{
        {3, 1}, {5, 1}, {5, 1}, {5, 1}, {7, 1}};

    vector_type A(A_data.size(), layout);
    vector_type B(B_data.size(), layout);
    vector_type D(A_data.size() + B_data.size(), layout);

    assign_vector(A, A_data);
    assign_vector(B, B_data);

    auto comp = [](tagged_value const& lhs, tagged_value const& rhs) {
        return lhs.key < rhs.key;
    };

    std::vector<tagged_value> expected;
    expected.reserve(D.size());
    std::merge(A_data.begin(), A_data.end(), B_data.begin(), B_data.end(),
        std::back_inserter(expected), comp);

    hpx::merge(hpx::execution::seq, A.begin(), A.end(), B.begin(), B.end(),
        D.begin(), comp);

    verify_merge(D, expected, "test_merge_boundary_duplicates");
}

///////////////////////////////////////////////////////////////////////////////
// Test 19: Larger data set — stress test with many elements
void test_merge_larger_data()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    auto layout = hpx::container_layout(localities);

    // Generate sorted data: A = even numbers, B = odd numbers
    std::size_t const N_A = 100;
    std::size_t const N_B = 80;

    std::vector<int> A_data(N_A);
    std::vector<int> B_data(N_B);

    for (std::size_t i = 0; i < N_A; ++i)
        A_data[i] = static_cast<int>(2 * i);    // 0, 2, 4, ..., 198

    for (std::size_t i = 0; i < N_B; ++i)
        B_data[i] = static_cast<int>(2 * i + 1);    // 1, 3, 5, ..., 159

    hpx::partitioned_vector<int> A(N_A, layout);
    hpx::partitioned_vector<int> B(N_B, layout);
    hpx::partitioned_vector<int> D(N_A + N_B, layout);

    assign_vector(A, A_data);
    assign_vector(B, B_data);

    std::vector<int> expected;
    expected.reserve(N_A + N_B);
    std::merge(A_data.begin(), A_data.end(), B_data.begin(), B_data.end(),
        std::back_inserter(expected));

    hpx::merge(
        hpx::execution::seq, A.begin(), A.end(), B.begin(), B.end(), D.begin());

    verify_merge(D, expected, "test_merge_larger_data");
}

///////////////////////////////////////////////////////////////////////////////
// Test 20: Larger data with non-contiguous partitions (multi-segment stress)
void test_merge_larger_noncontiguous()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    if (localities.size() < 2)
        return;

    // A: 60 elements in 3 partitions (20 each) -> [loc0, loc1, loc0]
    std::size_t const N_A = 60;
    std::vector<int> A_data(N_A);
    for (std::size_t i = 0; i < N_A; ++i)
        A_data[i] = static_cast<int>(i * 3);    // 0, 3, 6, ..., 177

    // B: 40 elements in 2 partitions (20 each) -> [loc0, loc1]
    std::size_t const N_B = 40;
    std::vector<int> B_data(N_B);
    for (std::size_t i = 0; i < N_B; ++i)
        B_data[i] = static_cast<int>(i * 3 + 1);    // 1, 4, 7, ..., 118

    // D: 100 elements in 3 partitions -> [loc0, loc1, loc0]
    // D sizes must divide: we need to pick something safe
    // 100 / 3 doesn't divide evenly, so use 4 partitions: 100/4 = 25
    hpx::partitioned_vector<int> A(N_A, hpx::container_layout(3, localities));
    hpx::partitioned_vector<int> B(N_B, hpx::container_layout(localities));
    hpx::partitioned_vector<int> D(
        N_A + N_B, hpx::container_layout(4, localities));

    assign_vector(A, A_data);
    assign_vector(B, B_data);

    std::vector<int> expected;
    expected.reserve(N_A + N_B);
    std::merge(A_data.begin(), A_data.end(), B_data.begin(), B_data.end(),
        std::back_inserter(expected));

    hpx::merge(
        hpx::execution::seq, A.begin(), A.end(), B.begin(), B.end(), D.begin());

    verify_merge(D, expected, "test_merge_larger_noncontiguous");
}

///////////////////////////////////////////////////////////////////////////////
// Test 21: par(task) with non-contiguous partitions
// Ensures the async return path works with complex layouts.
void test_merge_async_noncontiguous()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    if (localities.size() < 2)
        return;

    std::vector<int> const A_data{1, 4, 7, 10, 13, 16};
    std::vector<int> const B_data{2, 5, 8, 11, 14, 17};

    // A: 3 partitions [loc0, loc1, loc0], B: 2 partitions, D: 3 partitions
    hpx::partitioned_vector<int> A(
        A_data.size(), hpx::container_layout(3, localities));
    hpx::partitioned_vector<int> B(
        B_data.size(), hpx::container_layout(localities));
    hpx::partitioned_vector<int> D(
        A_data.size() + B_data.size(), hpx::container_layout(3, localities));

    assign_vector(A, A_data);
    assign_vector(B, B_data);

    std::vector<int> expected;
    expected.reserve(D.size());
    std::merge(A_data.begin(), A_data.end(), B_data.begin(), B_data.end(),
        std::back_inserter(expected));

    auto result = hpx::merge(hpx::execution::par(hpx::execution::task),
        A.begin(), A.end(), B.begin(), B.end(), D.begin());

    HPX_TEST(result.get() == D.end());
    verify_merge(D, expected, "test_merge_async_noncontiguous");
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    using namespace hpx::execution;

    // Basic tests (all locality counts)
    test_merge_no_policy();
    test_merge_stability(seq);
    test_merge_stability(par);
    test_merge_stability(par(task));
    test_merge_mixed_types(seq);
    test_merge_mixed_types(par(task));
    test_merge_empty_inputs(seq);
    test_merge_empty_inputs(par(task));
    test_merge_both_empty();
    test_merge_single_elements();
    test_merge_all_duplicates();
    test_merge_nonoverlapping_A_less();
    test_merge_nonoverlapping_A_greater();
    test_merge_skewed_sizes();
    test_merge_default_comp();
    test_merge_boundary_duplicates();
    test_merge_larger_data();

    // Multi-locality tests (require 2+ localities, skip on single)
    test_merge_noncontiguous_A();
    test_merge_noncontiguous_D();
    test_merge_different_ownership();
    test_merge_A_and_D_on_loc0_B_on_loc1();
    test_merge_noncontiguous_A_and_D();
    test_merge_participant_no_input_slices();
    test_merge_larger_noncontiguous();
    test_merge_async_noncontiguous();

    return hpx::util::report_errors();
}

#endif
