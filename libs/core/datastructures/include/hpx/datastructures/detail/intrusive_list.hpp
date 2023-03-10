//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org/libs/intrusive for documentation.

#pragma once

#include <hpx/assert.hpp>

#include <cstddef>
#include <utility>

namespace hpx::detail {

    template <typename Entry>
    class intrusive_list
    {
    public:
        constexpr intrusive_list() noexcept = default;

        intrusive_list(intrusive_list const&) = delete;
        intrusive_list& operator=(intrusive_list const&) = delete;

        intrusive_list(intrusive_list&&) = default;
        intrusive_list& operator=(intrusive_list&&) = default;

        ~intrusive_list() = default;

        void push_back(Entry& e) noexcept
        {
            if (last_entry == nullptr)
            {
                HPX_ASSERT(num_entries == 0);
                HPX_ASSERT(root == nullptr);

                e.prev = nullptr;
                e.next = nullptr;

                root = &e;
                last_entry = &e;
            }
            else
            {
                HPX_ASSERT(num_entries != 0);
                HPX_ASSERT(root != nullptr);

                e.prev = last_entry;
                e.next = nullptr;

                last_entry->next = &e;
                last_entry = &e;
            }
            ++num_entries;
        }

        void pop_front() noexcept
        {
            HPX_ASSERT(num_entries != 0);

            --num_entries;
            if (root->next != nullptr)
            {
                root->next->prev = nullptr;
            }
            else
            {
                last_entry = nullptr;
            }
            root = root->next;
        }

        void splice(intrusive_list& queue) noexcept
        {
            Entry* start = queue.front();
            Entry* end = queue.back();

            if (start != nullptr)
            {
                start->prev = last_entry;
            }

            if (last_entry != nullptr)
            {
                last_entry->next = start;
                if (end != nullptr)
                {
                    last_entry = end;
                }
            }
            else
            {
                root = start;
                last_entry = end;
            }

            num_entries += queue.size();

            queue.reset();
        }

        void erase(Entry const* e) noexcept
        {
            HPX_ASSERT(num_entries != 0);
            if (e == nullptr)
                return;

            --num_entries;
            if (e->next != nullptr)
            {
                e->next->prev = e->prev;
            }
            else
            {
                last_entry = e->prev;
            }

            if (e->prev != nullptr)
            {
                e->prev->next = e->next;
            }
            else
            {
                root = e->next;
            }
        }

        void reset() noexcept
        {
            num_entries = 0;
            root = nullptr;
            last_entry = nullptr;
        }

        [[nodiscard]] Entry* front() noexcept
        {
            return root;
        }
        [[nodiscard]] constexpr Entry const* front() const noexcept
        {
            return root;
        }

        [[nodiscard]] Entry* back() noexcept
        {
            return last_entry;
        }
        [[nodiscard]] constexpr Entry const* back() const noexcept
        {
            return last_entry;
        }

        [[nodiscard]] constexpr std::size_t size() const noexcept
        {
            return num_entries;
        }

        [[nodiscard]] constexpr bool empty() const noexcept
        {
            return num_entries == 0;
        }

        void swap(intrusive_list& rhs) noexcept
        {
            std::swap(num_entries, rhs.num_entries);
            std::swap(root, rhs.root);
            std::swap(last_entry, rhs.last_entry);
        }

    private:
        std::size_t num_entries = 0;
        Entry* root = nullptr;
        Entry* last_entry = nullptr;
    };
}    // namespace hpx::detail
