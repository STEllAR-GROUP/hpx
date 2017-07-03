//  Copyright (c) 2017 Denis Blank
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/lcos/future.hpp>
#include <hpx/traits/future_traits.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/util/pack_traversal.hpp>

#include <algorithm>
#include <functional>
#include <list>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

using namespace hpx;
using namespace hpx::util;
using namespace hpx::util::detail;

struct all_map_float
{
    template <typename T>
    float operator()(T el) const
    {
        return float(el + 1.f);
    }
};

struct my_mapper
{
    template <typename T,
        typename std::enable_if<std::is_same<T, int>::value>::type* = nullptr>
    float operator()(T el) const
    {
        return float(el + 1.f);
    }
};

struct all_map
{
    template <typename T>
    int operator()(T el) const
    {
        return 0;
    }
};

static void testMixedTraversal()
{
    {
        auto res = map_pack(all_map_float{},
            0,
            1.f,
            hpx::util::make_tuple(1.f, 3),
            std::vector<std::vector<int>>{{1, 2}, {4, 5}},
            std::vector<std::vector<float>>{{1.f, 2.f}, {4.f, 5.f}},
            2);

        auto expected = hpx::util::make_tuple(    // ...
            1.f,
            2.f,
            hpx::util::make_tuple(2.f, 4.f),
            std::vector<std::vector<float>>{{2.f, 3.f}, {5.f, 6.f}},
            std::vector<std::vector<float>>{{2.f, 3.f}, {5.f, 6.f}},
            3.f);

        static_assert(std::is_same<decltype(res), decltype(expected)>::value,
            "Type mismatch!");
        HPX_TEST((res == expected));
    }

    {
        // Broken build regression tests:
        traverse_pack(my_mapper{}, int(0), 1.f);
        map_pack(all_map{}, 0, std::vector<int>{1, 2});
    }

    {
        // Also a regression test
        auto res = map_pack(all_map{}, std::vector<std::vector<int>>{{1, 2}});
        HPX_TEST_EQ((res[0][0]), (0));
    }

    {
        auto res = map_pack(my_mapper{},
            0,
            1.f,
            hpx::util::make_tuple(1.f, 3,
                std::vector<std::vector<int>>{{1, 2}, {4, 5}},
                std::vector<std::vector<float>>{{1.f, 2.f}, {4.f, 5.f}}),
            2);

        auto expected = hpx::util::make_tuple(    // ...
            1.f,
            1.f,
            hpx::util::make_tuple(1.f, 4.f,
                std::vector<std::vector<float>>{{2.f, 3.f}, {5.f, 6.f}},
                std::vector<std::vector<float>>{{1.f, 2.f}, {4.f, 5.f}}),
            3.f);

        static_assert(std::is_same<decltype(res), decltype(expected)>::value,
            "Type mismatch!");
        HPX_TEST((res == expected));
    }

    {
        int count = 0;
        traverse_pack(
            [&](int el) {
                HPX_TEST_EQ((el), (count + 1));
                count = el;
            },
            1,
            hpx::util::make_tuple(
                2, 3, std::vector<std::vector<int>>{{4, 5}, {6, 7}}));

        HPX_TEST_EQ((count), (7));
    }

    return;
}

struct my_unwrapper
{
    template <typename T,
        typename std::enable_if<traits::is_future<T>::value>::type* = nullptr>
    auto operator()(T future) const ->
        typename traits::future_traits<T>::result_type
    {
        return future.get();
    }
};

static void testMixedEarlyUnwrapped()
{
    using namespace hpx::lcos;

    {
        auto res = map_pack(my_unwrapper{},    // ...
            0, 1, make_ready_future(3),
            make_tuple(make_ready_future(4), make_ready_future(5)));

        auto expected =
            hpx::util::make_tuple(0, 1, 3, hpx::util::make_tuple(4, 5));

        static_assert(std::is_same<decltype(res), decltype(expected)>::value,
            "Type mismatch!");
        HPX_TEST((res == expected));
    }
}

template <typename T>
struct my_allocator
{
    using value_type = T;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    using pointer = T*;
    using const_pointer = T const*;
    using reference = T&;
    using const_reference = T const&;

    unsigned state_;

    explicit my_allocator(unsigned state)
      : state_(state)
    {
        return;
    }

    template <typename O>
    my_allocator(my_allocator<O> const& other)
      : state_(other.state_)
    {
        return;
    }

    template <typename O>
    my_allocator& operator=(my_allocator<O> const& other)
    {
        state_ = other.state_;
        return *this;
    }

    template <typename O>
    struct rebind
    {
        using other = my_allocator<O>;
    };

    pointer allocate(size_type n, void const* hint = nullptr)
    {
        return std::allocator<T>{}.allocate(n, hint);
    }

    void deallocate(pointer p, size_type n)
    {
        return std::allocator<T>{}.deallocate(p, n);
    }
};

static void testMixedContainerRemap()
{
    // Traits
    {
        using namespace container_remapping;
        HPX_TEST_EQ((has_push_back<std::vector<int>, int>::value), true);
        HPX_TEST_EQ((has_push_back<int, int>::value), false);
    }

    // Rebind
    {
        auto const remapper = [](unsigned short i) -> unsigned long {
            return i - 1;
        };

        // Rebinds the values
        {
            std::vector<unsigned short> source = {1, 2, 3};
            std::vector<unsigned long> dest = map_pack(remapper, source);

            HPX_TEST((dest == decltype(dest){0, 1, 2}));
        }

        // Rebinds the allocator
        {
            static unsigned const canary = 78787;

            my_allocator<unsigned short> allocator(canary);
            std::vector<unsigned short, my_allocator<unsigned short>> source(
                allocator);

            // Empty
            {
                std::vector<unsigned long, my_allocator<unsigned long>>
                    remapped = map_pack(remapper, source);

                HPX_TEST_EQ((remapped.get_allocator().state_), (canary));
            }

            // Non empty
            source.push_back(1);
            {
                std::vector<unsigned long, my_allocator<unsigned long>>
                    remapped = map_pack(remapper, source);

                HPX_TEST_EQ((remapped.get_allocator().state_), (canary));
            }
        }
    }
}

struct mytester
{
    using traversor_type = mytester;

    int operator()(int)
    {
        return 0;
    }
};

struct my_int_mapper
{
    template <typename T,
        typename std::enable_if<std::is_same<T, int>::value>::type* = nullptr>
    float operator()(T el) const
    {
        return float(el + 1.f);
    }
};

static void testMixedFallThrough()
{
    traverse_pack(my_int_mapper{}, int(0),
        std::vector<hpx::util::tuple<float, float>>{
            hpx::util::make_tuple(1.f, 2.f)},
        hpx::util::make_tuple(std::vector<float>{1.f, 2.f}));

    traverse_pack(my_int_mapper{}, int(0),
        std::vector<std::vector<float>>{{1.f, 2.f}},
        hpx::util::make_tuple(1.f, 2.f));

    auto res1 = map_pack(my_int_mapper{}, int(0),
        std::vector<std::vector<float>>{{1.f, 2.f}},
        hpx::util::make_tuple(77.f, 2));

    auto res2 = map_pack(
        [](int) {
            // ...
            return 0;
        },
        1, std::vector<int>{2, 3});
}

class counter_mapper
{
    std::reference_wrapper<int> counter_;

public:
    explicit counter_mapper(int& counter)
      : counter_(counter)
    {
    }

    template <typename T>
    void operator()(T el) const
    {
        ++counter_.get();
    }
};

struct test_tag_1
{
};
struct test_tag_2
{
};
struct test_tag_3
{
};

class counter_mapper_rejecting_non_tag_1
{
    std::reference_wrapper<int> counter_;

public:
    explicit counter_mapper_rejecting_non_tag_1(int& counter)
      : counter_(counter)
    {
    }

    void operator()(test_tag_1)
    {
        ++counter_.get();
    }
};

struct tag_shift_mapper
{
    test_tag_2 operator()(test_tag_1) const
    {
        return {};
    }

    test_tag_3 operator()(test_tag_2) const
    {
        return {};
    }

    test_tag_1 operator()(test_tag_3) const
    {
        return {};
    }

    float operator()(int) const
    {
        return 0.f;
    }
};

class counter_mapper_rejecting_non_tag_1_sfinae
{
    std::reference_wrapper<int> counter_;

public:
    explicit counter_mapper_rejecting_non_tag_1_sfinae(int& counter)
      : counter_(counter)
    {
    }

    template <typename T,
        typename std::enable_if<std::is_same<typename std::decay<T>::type,
            test_tag_1>::value>::type* = nullptr>
    void operator()(T)
    {
        ++counter_.get();
    }
};

static void testStrategicTraverse()
{
    // Every element in the pack is visited
    {
        int counter = 0;
        counter_mapper mapper(counter);
        traverse_pack(mapper, test_tag_1{}, test_tag_2{}, test_tag_3{});
        HPX_TEST_EQ(counter, 3);
    }

    // Every element in the pack is visited from left to right
    {
        int counter = 0;
        traverse_pack(
            [&](int el) {
                HPX_TEST_EQ(counter, el);
                ++counter;
            },
            0, 1, 2, 3);
        HPX_TEST_EQ(counter, 4);
    }

    // Elements accepted by the mapper aren't traversed:
    // - Signature
    {
        int counter = 0;
        counter_mapper_rejecting_non_tag_1 mapper(counter);
        traverse_pack(mapper, test_tag_1{}, test_tag_2{}, test_tag_3{});
        HPX_TEST_EQ(counter, 1);
    }

    // - SFINAE
    {
        int counter = 0;
        counter_mapper_rejecting_non_tag_1_sfinae mapper(counter);
        traverse_pack(mapper, test_tag_1{}, test_tag_2{}, test_tag_3{});
        HPX_TEST_EQ(counter, 1);
    }

    // Remapping works across values
    {
        tuple<int, int, int> res =
            map_pack([](int i) { return i + 1; }, 0, 1, 2);

        auto expected = make_tuple(1, 2, 3);
        HPX_TEST((res == expected));
    }

    // Remapping works across types
    {
        tag_shift_mapper mapper;
        tuple<float, test_tag_2, test_tag_3, test_tag_1> res =
            map_pack(mapper, 1, test_tag_1{}, test_tag_2{}, test_tag_3{});

        HPX_TEST_EQ(get<0>(res), 0.f);
    }

    // Remapping works with move-only objects
    {
        std::unique_ptr<int> p1(new int(1));
        std::unique_ptr<int> p2(new int(2));
        std::unique_ptr<int> p3(new int(3));

        tuple<std::unique_ptr<unsigned>, std::unique_ptr<unsigned>,
            std::unique_ptr<unsigned>>
            res = map_pack(
                // Since we pass the unique_ptr's as r-value,
                // those should be passed as r-values to the mapper.
                [](std::unique_ptr<int>&& ptr) {
                    // We explicitly move the ownership here
                    std::unique_ptr<int> owned = std::move(ptr);
                    return std::unique_ptr<unsigned>(new unsigned(*owned + 1));
                },
                std::move(p1), std::move(p2), std::move(p3));

        // We expect the ownership of p1 - p3 to be invalid
        HPX_TEST((!bool(p1)));
        HPX_TEST((!bool(p2)));
        HPX_TEST((!bool(p3)));

        HPX_TEST_EQ((*get<0>(res)), 2U);
        HPX_TEST_EQ((*get<1>(res)), 3U);
        HPX_TEST_EQ((*get<2>(res)), 4U);
    }

    // Single object remapping returns the value itself without any boxing
    {
        int res = map_pack([](int i) { return i; }, 1);
        HPX_TEST_EQ(res, 1);
    }
}

static void testStrategicContainerTraverse()
{
    // Every element in the container is visited
    // - Plain container
    {
        int counter = 0;
        counter_mapper mapper(counter);
        std::vector<int> container;
        container.resize(100);
        traverse_pack(mapper, std::move(container));
        HPX_TEST_EQ(counter, 100);
    }

    // - Nested container
    {
        int counter = 0;
        counter_mapper mapper(counter);
        std::vector<std::vector<int>> container;
        for (unsigned i = 0; i < 10; ++i)
        {
            std::vector<int> nested;
            nested.resize(10);
            container.push_back(nested);
        }

        traverse_pack(mapper, std::move(container));
        HPX_TEST_EQ(counter, 100);
    }

    // Every element in the container is visited from left to right
    {
        int counter = 0;
        traverse_pack(
            [&](int el) {
                HPX_TEST_EQ(counter, el);
                ++counter;
            },
            std::vector<int>{0, 1},
            std::vector<std::vector<int>>{{2, 3}, {4, 5}});
        HPX_TEST_EQ(counter, 6);
    }

    // The container type itself is changed
    // - Plain container
    {
        std::vector<int> container{1, 2, 3};
        std::vector<float> res =
            map_pack([](int) { return 0.f; }, std::move(container));
        HPX_TEST_EQ(res.size(), 3U);
    }

    // - Nested container
    {
        std::vector<std::vector<int>> container;
        std::vector<std::vector<float>> res =
            map_pack([](int) { return 0.f; }, std::move(container));
    }

    // - Move only container
    {
        std::vector<std::unique_ptr<int>> container;
        container.push_back(std::unique_ptr<int>(new int(5)));
        std::vector<int> res =
            map_pack([](std::unique_ptr<int>&& ptr) { return *ptr; },
                std::move(container));

        HPX_TEST_EQ(res.size(), 1U);
        HPX_TEST_EQ(res[0], 5);
    }

    // Every element in the container is remapped
    // - Plain container
    {
        std::vector<int> container(100, 1);
        auto res = map_pack([](int i) { return 2; }, std::move(container));

        HPX_TEST((
            std::all_of(res.begin(), res.end(), [](int i) { return i == 2; })));
    }

    // - Nested container
    {
        std::vector<std::list<int>> container;
        for (unsigned i = 0; i < 10; ++i)
        {
            std::list<int> nested(10, 1);
            container.push_back(nested);
        }

        auto res = map_pack([](int i) { return 2; }, std::move(container));
        HPX_TEST((std::all_of(
            res.begin(), res.end(), [](std::list<int> const& nested) {
                return std::all_of(
                    nested.begin(), nested.end(), [](int i) { return i == 2; });
            })));
    }
}

static void testStrategicTupleLikeTraverse()
{
    // Every element in the tuple like type is visited
    {
        int counter = 0;
        counter_mapper mapper(counter);
        traverse_pack(
            mapper, make_tuple(test_tag_1{}, test_tag_2{}, test_tag_3{}));
        HPX_TEST_EQ(counter, 3);
    }

    // Every element in the tuple like type is visited from left to right
    {
        int counter = 0;
        traverse_pack(
            [&](int el) {
                HPX_TEST_EQ(counter, el);
                ++counter;
            },
            make_tuple(0, 1),
            make_tuple(make_tuple(2, 3), make_tuple(4, 5)),
            make_tuple(make_tuple(make_tuple(6, 7))));
        HPX_TEST_EQ(counter, 8);
    }

    // The container tuple like type itself is changed
    {
        tag_shift_mapper mapper;
        tuple<float, test_tag_2, test_tag_3, test_tag_1> res = map_pack(
            mapper, make_tuple(1, test_tag_1{}, test_tag_2{}, test_tag_3{}));

        HPX_TEST_EQ(get<0>(res), 0.f);
    }

    // Every element in the tuple like type is remapped
    {
        tuple<float, float, float> res =
            map_pack([](int) { return 1.f; }, make_tuple(0, 0, 0));

        auto expected = make_tuple(1.f, 1.f, 1.f);

        static_assert(std::is_same<decltype(res), decltype(expected)>::value,
            "Type mismatch!");
        HPX_TEST((res == expected));
    }
}

int main(int, char**)
{
    testMixedTraversal();
    testMixedEarlyUnwrapped();
    testMixedContainerRemap();
    testMixedFallThrough();

    testStrategicTraverse();
    testStrategicContainerTraverse();
    testStrategicTupleLikeTraverse();

    return hpx::util::report_errors();
}
