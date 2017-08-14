//  Copyright (c) 2017 Denis Blank
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/util/pack_traversal_async.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/unused.hpp>

#include <boost/smart_ptr/intrusive_ref_counter.hpp>

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <set>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(HPX_HAVE_CXX11_STD_ARRAY)
#include <array>
#endif

using hpx::util::async_traverse_complete_tag;
using hpx::util::async_traverse_detach_tag;
using hpx::util::async_traverse_visit_tag;
using hpx::util::make_tuple;
using hpx::util::traverse_pack_async;
using hpx::util::tuple;

template <typename Child>
class async_counter_base : public boost::intrusive_ref_counter<Child>
{
    std::size_t counter_ = 0;

public:
    explicit async_counter_base() = default;

    std::size_t const& counter() const noexcept
    {
        return counter_;
    }

    std::size_t& counter() noexcept
    {
        return counter_;
    }
};

template <std::size_t ArgCount>
struct async_increasing_int_sync_visitor
  : async_counter_base<async_increasing_int_sync_visitor<ArgCount>>
{
    bool operator()(async_traverse_visit_tag, std::size_t i)
    {
        HPX_TEST_EQ(i, this->counter());
        ++this->counter();
        return true;
    }

    template <typename N>
    void operator()(async_traverse_detach_tag, std::size_t i, N&& next)
    {
        HPX_UNUSED(i);
        HPX_UNUSED(next);

        // Should never be called!
        HPX_TEST(false);
    }

    template <typename T>
    void operator()(async_traverse_complete_tag, T&& pack)
    {
        HPX_UNUSED(pack);

        HPX_TEST_EQ(this->counter(), ArgCount);
        ++this->counter();
    }
};

template <std::size_t ArgCount>
struct async_increasing_int_visitor
  : async_counter_base<async_increasing_int_visitor<ArgCount>>
{
    bool operator()(async_traverse_visit_tag, std::size_t i) const
    {
        HPX_TEST_EQ(i, this->counter());
        return false;
    }

    template <typename N>
    void operator()(async_traverse_detach_tag, std::size_t i, N&& next)
    {
        HPX_UNUSED(i);

        ++this->counter();
        std::forward<N>(next)();
    }

    template <typename T>
    void operator()(async_traverse_complete_tag, T&& pack)
    {
        HPX_UNUSED(pack);

        HPX_TEST_EQ(this->counter(), ArgCount);
        ++this->counter();
    }
};

template <std::size_t ArgCount>
struct async_increasing_int_interrupted_visitor
  : async_counter_base<async_increasing_int_interrupted_visitor<ArgCount>>
{
    bool operator()(async_traverse_visit_tag, std::size_t i)
    {
        HPX_TEST_EQ(i, this->counter());
        ++this->counter();

        // Detach the control flow at the second step
        return i == 0;
    }

    template <typename N>
    void operator()(async_traverse_detach_tag, std::size_t i, N&& next)
    {
        HPX_TEST_EQ(i, 1U);
        HPX_TEST_EQ(this->counter(), 2U);

        // Don't call next here
        HPX_UNUSED(next);
    }

    template <typename T>
    void operator()(async_traverse_complete_tag, T&& pack) const
    {
        HPX_UNUSED(pack);

        // Will never be called
        HPX_TEST(false);
    }
};

template <std::size_t ArgCount, typename... Args>
void test_async_traversal_base(Args&&... args)
{
    // Test that every element is traversed in the correct order
    // when we detach the control flow on every visit.
    {
        auto result = traverse_pack_async(
            async_increasing_int_sync_visitor<ArgCount>{}, args...);
        HPX_TEST_EQ(result->counter(), ArgCount + 1U);
    }

    // Test that every element is traversed in the correct order
    // when we detach the control flow on every visit.
    {
        auto result = traverse_pack_async(
            async_increasing_int_visitor<ArgCount>{}, args...);
        HPX_TEST_EQ(result->counter(), ArgCount + 1U);
    }

    // Test that the first element is traversed only,
    // if we don't call the resume continuation.
    {
        auto result = traverse_pack_async(
            async_increasing_int_interrupted_visitor<ArgCount>{}, args...);
        HPX_TEST_EQ(result->counter(), 2U);
    }
}

static void test_async_traversal()
{
    // Just test everything using a casual int pack
    test_async_traversal_base<4U>(0U, 1U, 2U, 3U);
}

template <typename ContainerFactory>
void test_async_container_traversal_impl(ContainerFactory&& container_of)
{
    // Test by passing a containers in the middle
    test_async_traversal_base<4U>(0U, container_of(1U, 2U), 3U);
    // Test by splitting the pack in two containers
    test_async_traversal_base<4U>(container_of(0U, 1U), container_of(2U, 3U));
    // Test by passing a huge containers to the traversal
    test_async_traversal_base<4U>(container_of(0U, 1U, 2U, 3U));
}

template <typename T>
struct common_container_factory
{
    template <typename... Args>
    T operator()(Args&&... args)
    {
        return T{std::forward<Args>(args)...};
    }
};

#if defined(HPX_HAVE_CXX11_STD_ARRAY)
template <typename T>
struct array_container_factory
{
    template <typename... Args, typename Array = std::array<T, sizeof...(Args)>>
    Array operator()(Args&&... args)
    {
        return Array{{std::forward<Args>(args)...}};
    }
};
#endif

static void test_async_container_traversal()
{
    {
        common_container_factory<std::vector<std::size_t>> factory;
        test_async_container_traversal_impl(factory);
    }

    {
        common_container_factory<std::list<std::size_t>> factory;
        test_async_container_traversal_impl(factory);
    }

    {
        common_container_factory<std::set<std::size_t>> factory;
        test_async_container_traversal_impl(factory);
    }

#if defined(HPX_HAVE_CXX11_STD_ARRAY)
    {
        array_container_factory<std::size_t> factory;
        test_async_container_traversal_impl(factory);
    }
#endif
}

static void test_async_tuple_like_traversal()
{
    // Test by passing a tuple in the middle
    test_async_traversal_base<4U>(0U, make_tuple(1U, 2U), 3U);
    // Test by splitting the pack in two tuples
    test_async_traversal_base<4U>(make_tuple(0U, 1U), make_tuple(2U, 3U));
    // Test by passing a huge tuple to the traversal
    test_async_traversal_base<4U>(make_tuple(0U, 1U, 2U, 3U));
}

template <typename T,
    typename... Args,
    typename Vector = std::vector<typename std::decay<T>::type>>
Vector vector_of(T&& first, Args&&... args)
{
    return Vector{std::forward<T>(first), std::forward<Args>(args)...};
}

static void test_async_mixed_traversal()
{
    using container_t = std::vector<std::size_t>;

    // Test hierarchies where container and tuple like types are mixed
    test_async_traversal_base<4U>(
        0U, hpx::util::make_tuple(container_t{1U, 2U}), 3U);

    test_async_traversal_base<4U>(
        hpx::util::make_tuple(0U, vector_of(vector_of(1U))),
        make_tuple(2U, 3U));

    test_async_traversal_base<4U>(
        vector_of(vector_of(make_tuple(0U, 1U, 2U, 3U))));
}

template <std::size_t ArgCount>
struct async_unique_sync_visitor
  : async_counter_base<async_unique_sync_visitor<ArgCount>>
{
    bool operator()(async_traverse_visit_tag, std::unique_ptr<std::size_t>& i)
    {
        HPX_TEST_EQ(*i, this->counter());
        ++this->counter();
        return true;
    }

    template <typename N>
    void operator()(async_traverse_detach_tag,
        std::unique_ptr<std::size_t>& i,
        N&& next)
    {
        HPX_UNUSED(i);
        HPX_UNUSED(next);

        // Should never be called!
        HPX_TEST(false);
    }

    template <typename T>
    void operator()(async_traverse_complete_tag, T&& pack)
    {
        HPX_UNUSED(pack);

        HPX_TEST_EQ(this->counter(), ArgCount);
        ++this->counter();
    }
};

template <std::size_t ArgCount>
struct async_unique_visitor : async_counter_base<async_unique_visitor<ArgCount>>
{
    bool operator()(async_traverse_visit_tag,
        std::unique_ptr<std::size_t>& i) const
    {
        HPX_TEST_EQ(*i, this->counter());
        return false;
    }

    template <typename N>
    void operator()(async_traverse_detach_tag,
        std::unique_ptr<std::size_t>& i,
        N&& next)
    {
        HPX_UNUSED(i);

        ++this->counter();
        std::forward<N>(next)();
    }

    template <typename T>
    void operator()(async_traverse_complete_tag, T&& pack)
    {
        HPX_UNUSED(pack);

        HPX_TEST_EQ(this->counter(), ArgCount);
        ++this->counter();
    }
};

static void test_async_move_only_traversal()
{
    auto const of = [](std::size_t i) {
        return std::unique_ptr<std::size_t>(new std::size_t(i));
    };

    {
        auto result = traverse_pack_async(
            async_unique_sync_visitor<4>{}, of(0), of(1), of(2), of(3));
        HPX_TEST_EQ(result->counter(), 5U);
    }

    {
        auto result = traverse_pack_async(
            async_unique_visitor<4>{}, of(0), of(1), of(2), of(3));
        HPX_TEST_EQ(result->counter(), 5U);
    }
}

int main(int, char**)
{
    test_async_traversal();
    test_async_container_traversal();
    test_async_tuple_like_traversal();
    test_async_mixed_traversal();
    test_async_move_only_traversal();

    return hpx::util::report_errors();
}
