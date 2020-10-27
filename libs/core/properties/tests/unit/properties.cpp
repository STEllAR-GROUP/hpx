#include <hpx/modules/properties.hpp>
#include <hpx/modules/testing.hpp>

struct requirable_preferable_property
{
    int v = 0;

    template <typename T>
    struct is_applicable_property : std::false_type
    {
    };

    template <typename T>
    static constexpr bool is_applicable_property_v =
        is_applicable_property<T>::value;

    static constexpr bool is_requirable = true;
    static constexpr bool is_preferable = true;
};

struct requirable_preferable_property2
{
    template <typename T>
    struct is_applicable_property : std::false_type
    {
    };

    template <typename T>
    static constexpr bool is_applicable_property_v =
        is_applicable_property<T>::value;

    static constexpr bool is_requirable = true;
    static constexpr bool is_preferable = true;
};

struct requirable_property
{
    int v = 0;

    template <typename T>
    struct is_applicable_property : std::false_type
    {
    };

    template <typename T>
    static constexpr bool is_applicable_property_v =
        is_applicable_property<T>::value;

    static constexpr bool is_requirable = true;
    static constexpr bool is_preferable = false;
};

struct nonapplicable_property
{
    int v = 0;

    template <typename T>
    struct is_applicable_property : std::false_type
    {
    };

    template <typename T>
    static constexpr bool is_applicable_property_v =
        is_applicable_property<T>::value;

    static constexpr bool is_requirable = true;
    static constexpr bool is_preferable = true;
};

struct requirable_concept_property
{
    int v = 0;

    template <typename T>
    struct is_applicable_property : std::false_type
    {
    };

    template <typename T>
    static constexpr bool is_applicable_property_v =
        is_applicable_property<T>::value;

    static constexpr bool is_requirable_concept = true;
    static constexpr bool is_requirable = false;
    static constexpr bool is_preferable = false;
};

struct statically_requirable_preferable_property
{
    template <typename T>
    struct is_applicable_property : std::false_type
    {
    };

    template <typename T>
    static constexpr bool is_applicable_property_v =
        is_applicable_property<T>::value;

    static constexpr bool is_requirable = true;
    static constexpr bool is_preferable = true;

    template <typename T,
        typename = typename std::enable_if<is_applicable_property_v<T>>::type>
    static constexpr int static_query_v = 42;

    static constexpr int value()
    {
        return 42;
    }
};

static std::size_t require_requirable_preferable_property_invocation_count = 0;

struct type1
{
    requirable_preferable_property rpp{};
    requirable_property rp{};
    nonapplicable_property np{};
    requirable_concept_property rcp{};

    type1 require(requirable_preferable_property p)
    {
        ++require_requirable_preferable_property_invocation_count;
        return {p, rp, np, rcp};
    }

    int query(requirable_preferable_property)
    {
        return rpp.v;
    }

    type1 require_concept(requirable_concept_property p)
    {
        return {rpp, rp, np, p};
    }
};

struct type2
{
};

template <>
struct requirable_preferable_property::is_applicable_property<type1>
  : std::true_type
{
};

template <>
struct requirable_preferable_property2::is_applicable_property<type2>
  : std::true_type
{
};

template <>
struct requirable_property::is_applicable_property<type1> : std::true_type
{
};

template <>
struct requirable_concept_property::is_applicable_property<type1>
  : std::true_type
{
};

template <>
struct statically_requirable_preferable_property::is_applicable_property<type1>
  : std::true_type
{
};

static std::size_t
    tag_invoke_require_requirable_preferable_property_invocation_count = 0;

type1 tag_invoke(
    hpx::experimental::require_t, type1 t, requirable_preferable_property p)
{
    ++tag_invoke_require_requirable_preferable_property_invocation_count;
    return {p, t.rp, t.np, t.rcp};
}

type2 tag_invoke(
    hpx::experimental::require_t, type2 t, requirable_preferable_property p)
{
    return t;
}

static std::size_t tag_invoke_require_requirable_property_invocation_count = 0;

type1 tag_invoke(hpx::experimental::require_t, type1 t, requirable_property p)
{
    ++tag_invoke_require_requirable_property_invocation_count;
    return {t.rpp, p, t.np, t.rcp};
}

int main()
{
    // Check that properties are applicable
    static_assert(hpx::experimental::is_applicable_property_v<type1,
                      requirable_preferable_property>,
        "should be applicable");
    static_assert(
        hpx::experimental::is_applicable_property_v<type1, requirable_property>,
        "should be applicable");
    static_assert(!hpx::experimental::is_applicable_property_v<type1,
                      nonapplicable_property>,
        "should be applicable");
    static_assert(hpx::experimental::is_applicable_property_v<type1,
                      requirable_concept_property>,
        "should be applicable");
    static_assert(hpx::experimental::is_applicable_property_v<type1,
                      statically_requirable_preferable_property>,
        "should be applicable");

    // Check if we can require, prefer, or query properties
    static_assert(!hpx::experimental::can_require_concept<type1,
                      requirable_preferable_property>::value,
        "should be requirable");
    static_assert(hpx::experimental::can_require<type1,
                      requirable_preferable_property>::value,
        "should be requirable");
    static_assert(hpx::experimental::can_prefer<type1,
                      requirable_preferable_property>::value,
        "should be preferable");
    static_assert(hpx::experimental::can_query<type1,
                      requirable_preferable_property>::value,
        "should be queriable");

    static_assert(!hpx::experimental::can_require_concept<type1,
                      requirable_property>::value,
        "should be requirable");
    static_assert(
        hpx::experimental::can_require<type1, requirable_property>::value,
        "should be requirable");
    static_assert(
        !hpx::experimental::can_prefer<type1, requirable_property>::value,
        "should be preferable");
    static_assert(
        !hpx::experimental::can_query<type1, requirable_property>::value,
        "should be queriable");

    static_assert(!hpx::experimental::can_require_concept<type1,
                      nonapplicable_property>::value,
        "should be requirable");
    static_assert(
        !hpx::experimental::can_require<type1, nonapplicable_property>::value,
        "should be requirable");
    static_assert(
        !hpx::experimental::can_prefer<type1, nonapplicable_property>::value,
        "should be preferable");
    static_assert(
        !hpx::experimental::can_query<type1, nonapplicable_property>::value,
        "should be queriable");

    static_assert(hpx::experimental::can_require_concept<type1,
                      requirable_concept_property>::value,
        "should be requirable");
    static_assert(!hpx::experimental::can_require<type1,
                      requirable_concept_property>::value,
        "should be requirable");
    static_assert(!hpx::experimental::can_prefer<type1,
                      requirable_concept_property>::value,
        "should be preferable");
    static_assert(!hpx::experimental::can_query<type1,
                      requirable_concept_property>::value,
        "should be queriable");

    static_assert(!hpx::experimental::can_require_concept<type1,
                      statically_requirable_preferable_property>::value,
        "should be requirable");
    static_assert(hpx::experimental::can_require<type1,
                      statically_requirable_preferable_property>::value,
        "should be requirable");
    static_assert(hpx::experimental::can_prefer<type1,
                      statically_requirable_preferable_property>::value,
        "should be preferable");
    static_assert(hpx::experimental::can_query<type1,
                      statically_requirable_preferable_property>::value,
        "should be queriable");

    // Check that the various property functions actually can be called, and
    // that they have an effect
    type1 t1;
    type2 t2;
    requirable_preferable_property rpp;
    requirable_property rp;
    requirable_concept_property rcp;
    statically_requirable_preferable_property srpp;

    // Only requirable_preferable_property and
    // statically_requirable_preferable_property can be queried
    HPX_TEST_EQ(
        hpx::experimental::query(t1, requirable_preferable_property{}), 0);
    HPX_TEST_EQ(hpx::experimental::query(t1, rpp), 0);
    HPX_TEST_EQ(hpx::experimental::query(
                    t1, statically_requirable_preferable_property{}),
        42);
    HPX_TEST_EQ(hpx::experimental::query(t1, srpp), 42);
    HPX_TEST_EQ(t1.rp.v, 0);
    HPX_TEST_EQ(t1.rcp.v, 0);

    // Only requirable_preferable_property and requirable_property can be required
    HPX_TEST_EQ(hpx::experimental::query(hpx::experimental::require(t1,
                                             requirable_preferable_property{1}),
                    rpp),
        1);
    rpp = {2};
    HPX_TEST_EQ(
        hpx::experimental::query(hpx::experimental::require(t1, rpp), rpp), 2);
    HPX_TEST_EQ(hpx::experimental::require(t1, requirable_property{3}).rp.v, 3);
    rp = {3};
    HPX_TEST_EQ(hpx::experimental::require(t1, rp).rp.v, 3);

    // Multi-argument require
    type1 t1_2 = hpx::experimental::require(
        t1, requirable_preferable_property{4}, requirable_property{5});
    HPX_TEST_EQ(hpx::experimental::query(t1_2, rpp), 4);
    HPX_TEST_EQ(t1_2.rp.v, 5);
    type1 t1_3 =
        hpx::experimental::require(t1, requirable_preferable_property{6},
            requirable_property{7}, requirable_preferable_property{8});
    // The latter instance should be required last
    HPX_TEST_EQ(hpx::experimental::query(t1_3, rpp), 8);
    HPX_TEST_EQ(t1_3.rp.v, 7);

    // Only requirable_preferable_property can be preferred
    HPX_TEST_EQ(hpx::experimental::query(hpx::experimental::prefer(t1,
                                             requirable_preferable_property{9}),
                    rpp),
        9);
    rpp = {10};
    HPX_TEST_EQ(
        hpx::experimental::query(hpx::experimental::prefer(t1, rpp), rpp), 10);

    // requirable_preferable_property2 is preferable on type2, but has no effect
    hpx::experimental::prefer(type2{}, requirable_preferable_property2{});

    // Multi-argument prefer
    type1 t1_4 = hpx::experimental::prefer(t1,
        requirable_preferable_property{11}, requirable_preferable_property{12},
        requirable_preferable_property{13});
    // The last instance should be required last
    HPX_TEST_EQ(hpx::experimental::query(t1_4, rpp), 13);

    // Only requirable_concept_property be require_concepted
    HPX_TEST_EQ(
        hpx::experimental::require_concept(t1, requirable_concept_property{14})
            .rcp.v,
        14);
    rcp = {15};
    HPX_TEST_EQ(hpx::experimental::require_concept(t1, rcp).rcp.v, 15);

    // Both requirable_preferable_property and requirable_property have
    // require_t tag_invoke overloads, but only the latter tag_invoke overload
    // should be used because the member function should have precedence over
    // the tag_invoke overload.
    std::size_t
        require_requirable_preferable_property_current_invocation_count =
            require_requirable_preferable_property_invocation_count;
    std::size_t
        tag_invoke_require_requirable_preferable_property_current_invocation_count =
            tag_invoke_require_requirable_preferable_property_invocation_count;
    std::size_t
        tag_invoke_require_requirable_property_current_invocation_count =
            tag_invoke_require_requirable_property_invocation_count;
    hpx::experimental::require(t1, requirable_preferable_property{16});
    HPX_TEST_EQ(
        require_requirable_preferable_property_current_invocation_count + 1,
        require_requirable_preferable_property_invocation_count);
    HPX_TEST_EQ(
        tag_invoke_require_requirable_preferable_property_current_invocation_count,
        tag_invoke_require_requirable_preferable_property_invocation_count);

    hpx::experimental::require(t1, requirable_property{17});
    HPX_TEST_EQ(
        tag_invoke_require_requirable_property_current_invocation_count + 1,
        tag_invoke_require_requirable_property_invocation_count);

    return hpx::util::report_errors();
}
