//  Copyright (c) 2017 Denis Blank
//  Copyright Andrey Semashev 2007 - 2013.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/pack_traversal/pack_traversal_async.hpp>
#include <hpx/thread_support/atomic_count.hpp>
#include <hpx/type_support/unused.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <set>
#include <type_traits>
#include <utility>
#include <vector>

using hpx::make_tuple;
using hpx::tuple;
using hpx::util::async_traverse_complete_tag;
using hpx::util::async_traverse_detach_tag;
using hpx::util::async_traverse_visit_tag;
using hpx::util::traverse_pack_async;

/// A tag which isn't accepted by any mapper
struct not_accepted_tag
{
};

struct thread_safe_counter
{
    typedef hpx::util::atomic_count type;

    static unsigned int load(hpx::util::atomic_count const& counter) noexcept
    {
        return static_cast<unsigned int>(static_cast<long>(counter));
    }

    static void increment(hpx::util::atomic_count& counter) noexcept
    {
        ++counter;
    }

    static unsigned int decrement(hpx::util::atomic_count& counter) noexcept
    {
        return static_cast<unsigned int>(--counter);
    }
};

template <typename Derived, typename CounterPolicy = thread_safe_counter>
class intrusive_ref_counter;

template <typename Derived, typename CounterPolicy>
void intrusive_ptr_add_ref(
    intrusive_ref_counter<Derived, CounterPolicy> const* p) noexcept;

template <typename Derived, typename CounterPolicy>
void intrusive_ptr_release(
    intrusive_ref_counter<Derived, CounterPolicy> const* p) noexcept;

template <typename Derived, typename CounterPolicy>
class intrusive_ref_counter
{
private:
    typedef typename CounterPolicy::type counter_type;

    mutable counter_type ref_counter;

public:
    intrusive_ref_counter() noexcept
      : ref_counter(1)
    {
    }

    unsigned int use_count() const noexcept
    {
        return CounterPolicy::load(ref_counter);
    }

protected:
    ~intrusive_ref_counter() = default;

    friend void intrusive_ptr_add_ref<Derived, CounterPolicy>(
        intrusive_ref_counter<Derived, CounterPolicy> const* p) noexcept;

    friend void intrusive_ptr_release<Derived, CounterPolicy>(
        intrusive_ref_counter<Derived, CounterPolicy> const* p) noexcept;
};

template <typename Derived, typename CounterPolicy>
inline void intrusive_ptr_add_ref(
    intrusive_ref_counter<Derived, CounterPolicy> const* p) noexcept
{
    CounterPolicy::increment(p->ref_counter);
}

template <typename Derived, typename CounterPolicy>
inline void intrusive_ptr_release(
    intrusive_ref_counter<Derived, CounterPolicy> const* p) noexcept
{
    if (CounterPolicy::decrement(p->ref_counter) == 0)
        delete static_cast<Derived const*>(p);
}

template <typename Child>
class async_counter_base : public intrusive_ref_counter<Child>
{
    std::size_t counter_ = 0;

public:
    async_counter_base() = default;

    virtual ~async_counter_base() {}

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
    explicit async_increasing_int_sync_visitor(int) {}

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
    explicit async_increasing_int_visitor(int) {}

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

template <std::size_t ArgCount, typename... Args>
void test_async_traversal_base(Args&&... args)
{
    // Test that every element is traversed in the correct order
    // when we detach the control flow on every visit.
    {
        auto result = traverse_pack_async(
            hpx::util::async_traverse_in_place_tag<
                async_increasing_int_sync_visitor<ArgCount>>{},
            42, args...);
        HPX_TEST_EQ(result->counter(), ArgCount + 1U);
    }

    // Test that every element is traversed in the correct order
    // when we detach the control flow on every visit.
    {
        auto result =
            traverse_pack_async(hpx::util::async_traverse_in_place_tag<
                                    async_increasing_int_visitor<ArgCount>>{},
                42, args...);
        HPX_TEST_EQ(result->counter(), ArgCount + 1U);
    }
}

static void test_async_traversal()
{
    // Just test everything using a casual int pack
    test_async_traversal_base<4U>(not_accepted_tag{}, 0U, 1U,
        not_accepted_tag{}, 2U, 3U, not_accepted_tag{});
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

template <typename T>
struct array_container_factory
{
    template <typename... Args, typename Array = std::array<T, sizeof...(Args)>>
    Array operator()(Args&&... args)
    {
        return Array{{std::forward<Args>(args)...}};
    }
};

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

    {
        array_container_factory<std::size_t> factory;
        test_async_container_traversal_impl(factory);
    }
}

static void test_async_tuple_like_traversal()
{
    // Test by passing a tuple in the middle
    test_async_traversal_base<4U>(
        not_accepted_tag{}, 0U, make_tuple(1U, not_accepted_tag{}, 2U), 3U);
    // Test by splitting the pack in two tuples
    test_async_traversal_base<4U>(
        make_tuple(0U, not_accepted_tag{}, 1U), make_tuple(2U, 3U));
    // Test by passing a huge tuple to the traversal
    test_async_traversal_base<4U>(make_tuple(0U, 1U, 2U, 3U));
}

template <typename T, typename... Args,
    typename Vector = std::vector<typename std::decay<T>::type>>
Vector vector_of(T&& first, Args&&... args)
{
    return Vector{std::forward<T>(first), std::forward<Args>(args)...};
}

static void test_async_mixed_traversal()
{
    using container_t = std::vector<std::size_t>;

    // Test hierarchies where container and tuple like types are mixed
    test_async_traversal_base<4U>(0U, hpx::make_tuple(container_t{1U, 2U}), 3U);

    test_async_traversal_base<4U>(
        hpx::make_tuple(
            0U, vector_of(not_accepted_tag{}), vector_of(vector_of(1U))),
        make_tuple(2U, 3U));

    test_async_traversal_base<4U>(
        vector_of(vector_of(make_tuple(0U, 1U, 2U, 3U))));
}

template <std::size_t ArgCount>
struct async_unique_sync_visitor
  : async_counter_base<async_unique_sync_visitor<ArgCount>>
{
    explicit async_unique_sync_visitor(int) {}

    bool operator()(async_traverse_visit_tag, std::unique_ptr<std::size_t>& i)
    {
        HPX_TEST_EQ(*i, this->counter());
        ++this->counter();
        return true;
    }

    template <typename N>
    void operator()(
        async_traverse_detach_tag, std::unique_ptr<std::size_t>& i, N&& next)
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
    explicit async_unique_visitor(int) {}

    bool operator()(
        async_traverse_visit_tag, std::unique_ptr<std::size_t>& i) const
    {
        HPX_TEST_EQ(*i, this->counter());
        return false;
    }

    template <typename N>
    void operator()(
        async_traverse_detach_tag, std::unique_ptr<std::size_t>& i, N&& next)
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
        auto result =
            traverse_pack_async(hpx::util::async_traverse_in_place_tag<
                                    async_unique_sync_visitor<4>>{},
                42, of(0), of(1), of(2), of(3));
        HPX_TEST_EQ(result->counter(), 5U);
    }

    {
        auto result = traverse_pack_async(
            hpx::util::async_traverse_in_place_tag<async_unique_visitor<4>>{},
            42, of(0), of(1), of(2), of(3));
        HPX_TEST_EQ(result->counter(), 5U);
    }
}

struct invalidate_visitor : async_counter_base<invalidate_visitor>
{
    explicit invalidate_visitor(int) {}

    bool operator()(async_traverse_visit_tag, std::shared_ptr<int>& i) const
    {
        HPX_TEST_EQ(*i, 22);
        return false;
    }

    template <typename N>
    void operator()(
        async_traverse_detach_tag, std::shared_ptr<int>& i, N&& next)
    {
        HPX_UNUSED(i);

        std::forward<N>(next)();
    }

    // Test whether the passed pack was passed as r-value reference
    void operator()(
        async_traverse_complete_tag, tuple<std::shared_ptr<int>>&& pack) const
    {
        // Invalidate the moved object
        tuple<std::shared_ptr<int>> moved = std::move(pack);

        HPX_UNUSED(moved);
    }
};

// Check whether the arguments are invalidated (moved out) when called
static void test_async_complete_invalidation()
{
    auto value = std::make_shared<int>(22);

    auto frame = traverse_pack_async(
        hpx::util::async_traverse_in_place_tag<invalidate_visitor>{}, 42,
        value);

    HPX_TEST_EQ(value.use_count(), 1U);
}

int main(int, char**)
{
    test_async_traversal();
    test_async_container_traversal();
    test_async_tuple_like_traversal();
    test_async_mixed_traversal();
    test_async_move_only_traversal();
    test_async_complete_invalidation();

    return hpx::util::report_errors();
}
