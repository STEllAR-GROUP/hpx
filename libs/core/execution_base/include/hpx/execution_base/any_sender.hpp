//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/assert.hpp>
#include <hpx/errors/error.hpp>
#include <hpx/errors/throw_exception.hpp>
#include <hpx/execution_base/sender.hpp>

#include <cstddef>
#include <cstring>
#include <exception>
#include <memory>
#include <type_traits>
#include <utility>

namespace hpx::detail {
    template <typename T>
    struct empty_vtable
    {
        static_assert(
            sizeof(T) == 0, "No empty vtable type defined for given type T");
    };

    template <typename T>
    struct get_empty_vtable
    {
        using empty_vtable_type = typename empty_vtable<T>::type;
        static_assert(std::is_base_of_v<T, empty_vtable_type>,
            "Given empty vtable type should be a base of T");

        static T* call()
        {
            static empty_vtable_type empty;
            return &empty;
        }
    };

    template <typename Base, std::size_t EmbeddedStorageSize,
        std::size_t AlignmentSize = sizeof(void*)>
    class movable_sbo_storage
    {
    protected:
        using base_type = Base;
        static constexpr std::size_t embedded_storage_size =
            EmbeddedStorageSize;
        static constexpr std::size_t alignment_size = AlignmentSize;

        // The union has two members:
        // - embedded_storage: Embedded storage size array used for types that
        //   are at most embedded_storage_size bytes large, and require at most
        //   alignment_size alignment.
        // - heap_storage: A pointer to base_type that is used when objects
        //   don't fit in the embedded storage.
        union
        {
            std::aligned_storage_t<embedded_storage_size, alignment_size>
                embedded_storage;
            base_type* heap_storage;
        };
        base_type* object;

        // Returns true when it's safe to use the embedded storage, i.e.
        // when the size and alignment of Impl are small enough.
        template <typename Impl>
        static constexpr bool can_use_embedded_storage()
        {
            constexpr bool fits_storage =
                sizeof(std::decay_t<Impl>) <= embedded_storage_size;
            constexpr bool sufficiently_aligned =
                alignof(std::decay_t<Impl>) <= alignment_size;
            return fits_storage && sufficiently_aligned;
        }

        bool using_embedded_storage() const noexcept
        {
            return object ==
                reinterpret_cast<base_type const*>(&embedded_storage);
        }

        bool empty() const noexcept
        {
            return get().empty();
        }

        void release()
        {
            HPX_ASSERT(!empty());

            if (using_embedded_storage())
            {
                get().~base_type();
            }
            else
            {
                delete heap_storage;
                heap_storage = nullptr;
            }

            object = get_empty_vtable<base_type>::call();
        }

        void move_assign(movable_sbo_storage&& other) &

        {
            HPX_ASSERT(&other != this);
            HPX_ASSERT(empty());

            if (!other.empty())
            {
                if (other.using_embedded_storage())
                {
                    auto p = reinterpret_cast<base_type*>(&embedded_storage);
                    other.get().move_into(p);
                    object = p;
                }
                else
                {
                    heap_storage = other.heap_storage;
                    other.heap_storage = nullptr;
                    object = heap_storage;
                }

                other.object = get_empty_vtable<base_type>::call();
            }
        }

    public:
        movable_sbo_storage()
          : heap_storage(nullptr)
          , object(get_empty_vtable<base_type>::call())
        {
        }

        ~movable_sbo_storage()
        {
            if (!empty())
            {
                release();
            }
        }

        movable_sbo_storage(movable_sbo_storage&& other)
          : heap_storage(nullptr)
          , object(get_empty_vtable<base_type>::call())
        {
            move_assign(std::move(other));
        }

        movable_sbo_storage& operator=(movable_sbo_storage&& other)
        {
            if (&other != this)
            {
                if (!empty())
                {
                    release();
                }

                move_assign(std::move(other));
            }
            return *this;
        }

        movable_sbo_storage(movable_sbo_storage const&) = delete;
        movable_sbo_storage& operator=(movable_sbo_storage const&) = delete;

        base_type const& get() const noexcept
        {
            return *object;
        }

        base_type& get() noexcept
        {
            return *object;
        }

        template <typename Impl, typename... Ts>
        void store(Ts&&... ts)
        {
            if (!empty())
            {
                release();
            }

            if constexpr (can_use_embedded_storage<Impl>())
            {
                Impl* p = reinterpret_cast<Impl*>(&embedded_storage);
                new (p) Impl(std::forward<Ts>(ts)...);
                object = p;
            }
            else
            {
                heap_storage = new Impl(std::forward<Ts>(ts)...);
                object = heap_storage;
            }
        }
    };

    template <typename Base, std::size_t EmbeddedStorageSize,
        std::size_t AlignmentSize = sizeof(void*)>
    class copyable_sbo_storage
      : public movable_sbo_storage<Base, EmbeddedStorageSize, AlignmentSize>
    {
        using storage_base_type =
            movable_sbo_storage<Base, EmbeddedStorageSize, AlignmentSize>;

        using typename storage_base_type::base_type;

        using storage_base_type::embedded_storage;
        using storage_base_type::empty;
        using storage_base_type::heap_storage;
        using storage_base_type::object;
        using storage_base_type::release;
        using storage_base_type::using_embedded_storage;

        void copy_assign(copyable_sbo_storage const& other) &
        {
            HPX_ASSERT(&other != this);
            HPX_ASSERT(empty());

            if (!other.empty())
            {
                if (other.using_embedded_storage())
                {
                    base_type* p =
                        reinterpret_cast<base_type*>(&embedded_storage);
                    other.get().clone_into(p);
                    object = p;
                }
                else
                {
                    heap_storage = other.get().clone();
                    object = heap_storage;
                }
            }
        }

    public:
        using storage_base_type::get;
        using storage_base_type::store;

        copyable_sbo_storage() = default;
        copyable_sbo_storage(copyable_sbo_storage&&) = default;
        copyable_sbo_storage& operator=(copyable_sbo_storage&&) = default;

        copyable_sbo_storage(copyable_sbo_storage const& other)
          : storage_base_type()
        {
            copy_assign(other);
        }

        copyable_sbo_storage& operator=(copyable_sbo_storage const& other)
        {
            if (&other != this)
            {
                if (!empty())
                {
                    release();
                }
                copy_assign(other);
            }
            return *this;
        }
    };
}    // namespace hpx::detail

namespace hpx::execution::experimental {
    namespace detail {
        struct any_operation_state_base
        {
            virtual ~any_operation_state_base() = default;
            virtual bool empty() const noexcept
            {
                return false;
            }
            virtual void start() & noexcept = 0;
        };

        struct HPX_CORE_EXPORT empty_any_operation_state final
          : any_operation_state_base
        {
            bool empty() const noexcept override;
            void start() & noexcept override;
        };

        template <typename Sender, typename Receiver>
        struct any_operation_state_impl final : any_operation_state_base
        {
            std::decay_t<
                connect_result_t<std::decay_t<Sender>, std::decay_t<Receiver>>>
                operation_state;

            template <typename Sender_, typename Receiver_>
            any_operation_state_impl(Sender_&& sender, Receiver_&& receiver)
              : operation_state(hpx::execution::experimental::connect(
                    std::forward<Sender_>(sender),
                    std::forward<Receiver_>(receiver)))
            {
            }

            void start() & noexcept override
            {
                hpx::execution::experimental::start(operation_state);
            }
        };

        class HPX_CORE_EXPORT any_operation_state
        {
            using base_type = detail::any_operation_state_base;
            template <typename Sender, typename Receiver>
            using impl_type =
                detail::any_operation_state_impl<Sender, Receiver>;
            using storage_type =
                hpx::detail::movable_sbo_storage<base_type, 8 * sizeof(void*)>;

            storage_type storage{};

        public:
            template <typename Sender, typename Receiver>
            any_operation_state(Sender&& sender, Receiver&& receiver)
            {
                storage.template store<impl_type<Sender, Receiver>>(
                    std::forward<Sender>(sender),
                    std::forward<Receiver>(receiver));
            }

            ~any_operation_state() = default;
            any_operation_state(any_operation_state&&) = delete;
            any_operation_state(any_operation_state const&) = delete;
            any_operation_state& operator=(any_operation_state&&) = delete;
            any_operation_state& operator=(any_operation_state const&) = delete;

            HPX_CORE_EXPORT friend void tag_dispatch(
                hpx::execution::experimental::start_t,
                any_operation_state& os) noexcept;
        };

        template <typename... Ts>
        struct any_receiver_base
        {
            virtual ~any_receiver_base() = default;
            virtual void move_into(void* p) = 0;
            virtual void set_value(Ts... ts) && = 0;
            virtual void set_error(std::exception_ptr ep) && noexcept = 0;
            virtual void set_done() && noexcept = 0;
            virtual bool empty() const noexcept
            {
                return false;
            }
        };

        HPX_NORETURN HPX_CORE_EXPORT void throw_bad_any_call(
            char const* class_name, char const* function_name);

        template <typename... Ts>
        struct empty_any_receiver final : any_receiver_base<Ts...>
        {
            void move_into(void*) override
            {
                HPX_UNREACHABLE;
            }

            bool empty() const noexcept override
            {
                return true;
            }

            void set_value(Ts...) && override
            {
                throw_bad_any_call("any_receiver", "set_value");
            }

            HPX_NORETURN void set_error(std::exception_ptr) && noexcept override
            {
                throw_bad_any_call("any_receiver", "set_error");
            }

            HPX_NORETURN void set_done() && noexcept override
            {
                throw_bad_any_call("any_receiver", "set_done");
            }
        };

        template <typename Receiver, typename... Ts>
        struct any_receiver_impl final : any_receiver_base<Ts...>
        {
            std::decay_t<Receiver> receiver;

            template <typename Receiver_,
                typename = std::enable_if_t<!std::is_same_v<
                    std::decay_t<Receiver_>, any_receiver_impl>>>
            explicit any_receiver_impl(Receiver_&& receiver)
              : receiver(std::forward<Receiver_>(receiver))
            {
            }

            void move_into(void* p) override
            {
                new (p) any_receiver_impl(std::move(receiver));
            }

            void set_value(Ts... ts) && override
            {
                hpx::execution::experimental::set_value(
                    std::move(receiver), std::move(ts)...);
            }

            void set_error(std::exception_ptr ep) && noexcept override
            {
                hpx::execution::experimental::set_error(
                    std::move(receiver), std::move(ep));
            }

            void set_done() && noexcept override
            {
                hpx::execution::experimental::set_done(std::move(receiver));
            }
        };

        template <typename... Ts>
        class any_receiver
        {
            using base_type = detail::any_receiver_base<Ts...>;
            template <typename Receiver>
            using impl_type = detail::any_receiver_impl<Receiver, Ts...>;
            using storage_type =
                hpx::detail::movable_sbo_storage<base_type, 4 * sizeof(void*)>;

            storage_type storage{};

        public:
            template <typename Receiver,
                typename = std::enable_if_t<
                    !std::is_same_v<std::decay_t<Receiver>, any_receiver>>>
            explicit any_receiver(Receiver&& receiver)
            {
                storage.template store<impl_type<Receiver>>(
                    std::forward<Receiver>(receiver));
            }

            template <typename Receiver,
                typename = std::enable_if_t<
                    !std::is_same_v<std::decay_t<Receiver>, any_receiver>>>
            any_receiver& operator=(Receiver&& receiver)
            {
                storage.template store<impl_type<Receiver>>(
                    std::forward<Receiver>(receiver));
                return *this;
            }

            ~any_receiver() = default;
            any_receiver(any_receiver&&) = default;
            any_receiver(any_receiver const&) = delete;
            any_receiver& operator=(any_receiver&&) = default;
            any_receiver& operator=(any_receiver const&) = delete;

            friend void tag_dispatch(hpx::execution::experimental::set_value_t,
                any_receiver&& r, Ts... ts)
            {
                // We first move the storage to a temporary variable so that
                // this any_receiver is empty after this set_value. Doing
                // std::move(storage.get()).set_value(...) would leave us with a
                // non-empty any_receiver holding a moved-from receiver.
                auto moved_storage = std::move(r.storage);
                std::move(moved_storage.get()).set_value(std::move(ts)...);
            }

            friend void tag_dispatch(hpx::execution::experimental::set_error_t,
                any_receiver&& r, std::exception_ptr ep) noexcept
            {
                // We first move the storage to a temporary variable so that
                // this any_receiver is empty after this set_error. Doing
                // std::move(storage.get()).set_error(...) would leave us with a
                // non-empty any_receiver holding a moved-from receiver.
                auto moved_storage = std::move(r.storage);
                std::move(moved_storage.get()).set_error(std::move(ep));
            }

            friend void tag_dispatch(hpx::execution::experimental::set_done_t,
                any_receiver&& r) noexcept
            {
                // We first move the storage to a temporary variable so that
                // this any_receiver is empty after this set_done. Doing
                // std::move(storage.get()).set_done(...) would leave us with a
                // non-empty any_receiver holding a moved-from receiver.
                auto moved_storage = std::move(r.storage);
                std::move(moved_storage.get()).set_done();
            }
        };

        template <typename... Ts>
        struct unique_any_sender_base
        {
            virtual ~unique_any_sender_base() = default;
            virtual void move_into(void* p) = 0;
            virtual any_operation_state connect(
                any_receiver<Ts...>&& receiver) && = 0;
            virtual bool empty() const noexcept
            {
                return false;
            }
        };

        template <typename... Ts>
        struct any_sender_base : public unique_any_sender_base<Ts...>
        {
            virtual any_sender_base* clone() const = 0;
            virtual void clone_into(void* p) const = 0;
            using unique_any_sender_base<Ts...>::connect;
            virtual any_operation_state connect(
                any_receiver<Ts...>&& receiver) & = 0;
        };

        template <typename... Ts>
        struct empty_unique_any_sender final : unique_any_sender_base<Ts...>
        {
            void move_into(void*) override
            {
                HPX_UNREACHABLE;
            }

            bool empty() const noexcept override
            {
                return true;
            }

            HPX_NORETURN any_operation_state connect(any_receiver<Ts...>&&) &&
                override
            {
                throw_bad_any_call("unique_any_sender", "connect");
            }
        };

        template <typename... Ts>
        struct empty_any_sender final : any_sender_base<Ts...>
        {
            void move_into(void*) override
            {
                HPX_UNREACHABLE;
            }

            any_sender_base<Ts...>* clone() const override
            {
                HPX_UNREACHABLE;
            }

            void clone_into(void*) const override
            {
                HPX_UNREACHABLE;
            }

            bool empty() const noexcept override
            {
                return true;
            }

            HPX_NORETURN any_operation_state connect(any_receiver<Ts...>&&) &
                override
            {
                throw_bad_any_call("any_sender", "connect");
            }

            HPX_NORETURN any_operation_state connect(any_receiver<Ts...>&&) &&
                override
            {
                throw_bad_any_call("any_sender", "connect");
            }
        };

        template <typename Sender, typename... Ts>
        struct unique_any_sender_impl final : unique_any_sender_base<Ts...>
        {
            std::decay_t<Sender> sender;

            template <typename Sender_,
                typename = std::enable_if_t<!std::is_same_v<
                    std::decay_t<Sender_>, unique_any_sender_impl>>>
            explicit unique_any_sender_impl(Sender_&& sender)
              : sender(std::forward<Sender_>(sender))
            {
            }

            void move_into(void* p) override
            {
                new (p) unique_any_sender_impl(std::move(sender));
            }

            any_operation_state connect(any_receiver<Ts...>&& receiver) &&
                override
            {
                return any_operation_state{
                    std::move(sender), std::move(receiver)};
            }
        };

        template <typename Sender, typename... Ts>
        struct any_sender_impl final : any_sender_base<Ts...>
        {
            std::decay_t<Sender> sender;

            template <typename Sender_,
                typename = std::enable_if_t<
                    !std::is_same_v<std::decay_t<Sender_>, any_sender_impl>>>
            explicit any_sender_impl(Sender_&& sender)
              : sender(std::forward<Sender_>(sender))
            {
            }

            void move_into(void* p) override
            {
                new (p) any_sender_impl(std::move(sender));
            }

            any_sender_base<Ts...>* clone() const override
            {
                return new any_sender_impl(sender);
            }

            void clone_into(void* p) const override
            {
                new (p) any_sender_impl(sender);
            }

            any_operation_state connect(any_receiver<Ts...>&& receiver) &
                override
            {
                return any_operation_state{sender, std::move(receiver)};
            }

            any_operation_state connect(any_receiver<Ts...>&& receiver) &&
                override
            {
                return any_operation_state{
                    std::move(sender), std::move(receiver)};
            }
        };
    }    // namespace detail

    template <typename... Ts>
    class unique_any_sender
    {
        using base_type = detail::unique_any_sender_base<Ts...>;
        template <typename Sender>
        using impl_type = detail::unique_any_sender_impl<Sender, Ts...>;
        using storage_type =
            hpx::detail::movable_sbo_storage<base_type, 4 * sizeof(void*)>;

        storage_type storage{};

    public:
        unique_any_sender() = default;

        template <typename Sender,
            typename = std::enable_if_t<
                !std::is_same_v<std::decay_t<Sender>, unique_any_sender>>>
        explicit unique_any_sender(Sender&& sender)
        {
            storage.template store<impl_type<Sender>>(
                std::forward<Sender>(sender));
        }

        template <typename Sender,
            typename = std::enable_if_t<
                !std::is_same_v<std::decay_t<Sender>, unique_any_sender>>>
        unique_any_sender& operator=(Sender&& sender)
        {
            storage.template store<impl_type<Sender>>(
                std::forward<Sender>(sender));
            return *this;
        }

        ~unique_any_sender() = default;
        unique_any_sender(unique_any_sender&&) = default;
        unique_any_sender(unique_any_sender const&) = delete;
        unique_any_sender& operator=(unique_any_sender&&) = default;
        unique_any_sender& operator=(unique_any_sender const&) = delete;

        template <template <typename...> class Tuple,
            template <typename...> class Variant>
        using value_types = Variant<Tuple<Ts...>>;

        template <template <typename...> class Variant>
        using error_types = Variant<std::exception_ptr>;

        static constexpr bool sends_done = false;

        template <typename R>
        friend detail::any_operation_state tag_dispatch(
            hpx::execution::experimental::connect_t, unique_any_sender&& s,
            R&& r)
        {
            // We first move the storage to a temporary variable so that this
            // any_sender is empty after this connect. Doing
            // std::move(storage.get()).connect(...) would leave us with a
            // non-empty any_sender holding a moved-from sender.
            auto moved_storage = std::move(s.storage);
            return std::move(moved_storage.get())
                .connect(detail::any_receiver<Ts...>{std::forward<R>(r)});
        }
    };

    template <typename... Ts>
    class any_sender
    {
        using base_type = detail::any_sender_base<Ts...>;
        template <typename Sender>
        using impl_type = detail::any_sender_impl<Sender, Ts...>;
        using storage_type =
            hpx::detail::copyable_sbo_storage<base_type, 4 * sizeof(void*)>;

        storage_type storage{};

    public:
        any_sender() = default;

        template <typename Sender,
            typename = std::enable_if_t<
                !std::is_same_v<std::decay_t<Sender>, any_sender>>>
        explicit any_sender(Sender&& sender)
        {
            static_assert(std::is_copy_constructible_v<std::decay_t<Sender>>,
                "any_sender requires the given sender to be copy "
                "constructible. Ensure the used sender type is copy "
                "constructible or use unique_any_sender if you do not require "
                "copyability.");
            storage.template store<impl_type<Sender>>(
                std::forward<Sender>(sender));
        }

        template <typename Sender,
            typename = std::enable_if_t<
                !std::is_same_v<std::decay_t<Sender>, any_sender>>>
        any_sender& operator=(Sender&& sender)
        {
            static_assert(std::is_copy_constructible_v<std::decay_t<Sender>>,
                "any_sender requires the given sender to be copy "
                "constructible. Ensure the used sender type is copy "
                "constructible or use unique_any_sender if you do not require "
                "copyability.");
            storage.template store<impl_type<Sender>>(
                std::forward<Sender>(sender));
            return *this;
        }

        ~any_sender() = default;
        any_sender(any_sender&&) = default;
        any_sender(any_sender const&) = default;
        any_sender& operator=(any_sender&&) = default;
        any_sender& operator=(any_sender const&) = default;

        template <template <typename...> class Tuple,
            template <typename...> class Variant>
        using value_types = Variant<Tuple<Ts...>>;

        template <template <typename...> class Variant>
        using error_types = Variant<std::exception_ptr>;

        static constexpr bool sends_done = false;

        template <typename R>
        friend detail::any_operation_state tag_dispatch(
            hpx::execution::experimental::connect_t, any_sender& s, R&& r)
        {
            return s.storage.get().connect(
                detail::any_receiver<Ts...>{std::forward<R>(r)});
        }

        template <typename R>
        friend detail::any_operation_state tag_dispatch(
            hpx::execution::experimental::connect_t, any_sender&& s, R&& r)
        {
            // We first move the storage to a temporary variable so that this
            // any_sender is empty after this connect. Doing
            // std::move(storage.get()).connect(...) would leave us with a
            // non-empty any_sender holding a moved-from sender.
            auto moved_storage = std::move(s.storage);
            return std::move(moved_storage.get())
                .connect(detail::any_receiver<Ts...>{std::forward<R>(r)});
        }
    };
}    // namespace hpx::execution::experimental

namespace hpx::detail {
    template <>
    struct empty_vtable<
        hpx::execution::experimental::detail::any_operation_state_base>
    {
        using type =
            hpx::execution::experimental::detail::empty_any_operation_state;
    };

    template <typename... Ts>
    struct empty_vtable<
        hpx::execution::experimental::detail::any_receiver_base<Ts...>>
    {
        using type =
            hpx::execution::experimental::detail::empty_any_receiver<Ts...>;
    };

    template <typename... Ts>
    struct empty_vtable<
        hpx::execution::experimental::detail::unique_any_sender_base<Ts...>>
    {
        using type =
            hpx::execution::experimental::detail::empty_unique_any_sender<
                Ts...>;
    };

    template <typename... Ts>
    struct empty_vtable<
        hpx::execution::experimental::detail::any_sender_base<Ts...>>
    {
        using type =
            hpx::execution::experimental::detail::empty_any_sender<Ts...>;
    };
}    // namespace hpx::detail
