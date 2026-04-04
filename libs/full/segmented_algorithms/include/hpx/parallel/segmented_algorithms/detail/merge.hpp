//  Copyright (c) 2026 Abhishek Bansal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <hpx/actions_base/plain_action.hpp>
#include <hpx/assert.hpp>
#include <hpx/collectives/all_gather.hpp>
#include <hpx/collectives/all_to_all.hpp>
#include <hpx/collectives/argument_types.hpp>
#include <hpx/components_base/agas_interface.hpp>
#include <hpx/modules/async_base.hpp>
#include <hpx/modules/async_combinators.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/naming_base.hpp>
#include <hpx/modules/serialization.hpp>

#include <hpx/modules/synchronization.hpp>
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iterator>
#include <limits>
#include <list>
#include <map>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hpx::collectives {
    template <typename T>
    std::vector<std::decay_t<T>> all_gather(hpx::launch::sync_policy,
        char const*, T&&, num_sites_arg const, this_site_arg const,
        generation_arg const, root_site_arg const);

    template <typename T>
    std::vector<T> all_to_all(hpx::launch::sync_policy, char const*,
        std::vector<T>&&, num_sites_arg const, this_site_arg const,
        generation_arg const, root_site_arg const);
}    // namespace hpx::collectives

namespace hpx::parallel::detail {

    // =====================================================================
    // Segmented Merge
    // =====================================================================
    //
    // Merges two sorted distributed ranges (A, B) into a distributed
    // destination range (D) using the co-rank technique.
    //
    // Overview:
    //   Each range is composed of segments (slices) spread across
    //   localities. The coordinator decomposes the three ranges into
    //   globally-ordered slice metadata, then launches a kernel
    //   on every participating locality. The kernel proceeds in phases:
    //
    //   Phase 1  Startup barrier: ensures all handle registries are
    //            populated before any remote co-rank probes fire.
    //   Phase 2  Co-rank planning: for each local destination slice,
    //            binary-search (co-rank) determines which sub-ranges
    //            of A and B contribute to it. Successive slices narrow
    //            the search bounds for reduced remote fetches.
    //   Phase 3  Interval sharing: all localities exchange their
    //            computed intervals via all_gather.
    //   Phase 4  Payload packing: local input data is packed into
    //            per-peer batches (one flat buffer + descriptors per
    //            peer) to minimize serialization overhead.
    //   Phase 5  Payload exchange: two all_to_all calls ship A and B
    //            batches to the localities that own the destination.
    //   Phase 6  Local merge: each locality resolves received fragment
    //            descriptors to zero-copy pointers into the batch
    //            buffers, then performs a standard stable merge into
    //            its destination slices.
    //
    // Key types:
    //   slice_metadata     global descriptor for one segment
    //   owned_handle       local-only handle with actual iterators
    //   interval           co-rank result for one dest slice
    //   fragment_desc      lightweight header indexing into a batch
    //   payload_batch      wire type: descriptors + flat value buffer
    //   resolved_fragment  kernel-internal pointer+count view
    //
    // =====================================================================

    ///////////////////////////////////////////////////////////////////////////
    // slice_metadata: globally-shared descriptor for one contiguous segment
    // of a segmented range (A, B, or D).
    struct distributed_merge_slice_metadata
    {
        std::uint64_t slice_id = 0;
        std::uint64_t site_index = 0;
        std::uint64_t global_begin = 0;
        std::uint64_t global_end = 0;
        std::uint32_t locality_id = naming::invalid_locality_id;
    };

    ///////////////////////////////////////////////////////////////////////////
    // owned_handle: local-only handle holding the actual iterators for
    // a segment owned by this locality. Never serialized. Local iterators
    // are meaningless on remote localities.
    template <typename LocalIter>
    struct distributed_merge_owned_handle
    {
        std::uint64_t slice_id = 0;
        std::uint64_t global_begin = 0;
        std::uint64_t global_end = 0;
        LocalIter local_begin{};
    };

    ///////////////////////////////////////////////////////////////////////////
    // interval: co-rank planning result for one destination slice.
    struct distributed_merge_interval
    {
        std::uint64_t dest_slice_id = 0;
        std::uint64_t site_index = 0;
        std::uint64_t A_begin_rank = 0;
        std::uint64_t A_end_rank = 0;
        std::uint64_t B_begin_rank = 0;
        std::uint64_t B_end_rank = 0;
    };

    ///////////////////////////////////////////////////////////////////////////
    // Fragment descriptor: on the wire it indexes into the batch's shared
    // values buffer via (value_offset, value_count). After receiving, the
    // data pointer is resolved to point directly into the buffer.
    struct distributed_merge_fragment_desc
    {
        std::uint64_t dest_slice_id = 0;
        std::uint64_t input_global_begin = 0;
        std::uint64_t value_offset = 0;
        std::uint64_t value_count = 0;
    };

    // Payload batch: all fragments destined for one peer, packed into a
    // single flat values buffer with lightweight descriptors.
    template <typename T>
    struct distributed_merge_payload_batch
    {
        std::vector<distributed_merge_fragment_desc> fragments;
        std::vector<T> values;

    private:
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, unsigned)
        {
            ar & fragments & values;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // collective_context: manages unique basenames and generation counters
    // so concurrent distributed merge invocations never collide.
    struct distributed_merge_collective_ctx
    {
        std::string invocation_ns;
        std::unordered_map<std::string, std::uint64_t> generations;

        explicit distributed_merge_collective_ctx(std::string ns)
          : invocation_ns(HPX_MOVE(ns))
        {
        }

        [[nodiscard]] std::string basename(std::string const& phase) const
        {
            return invocation_ns + "/" + phase;
        }

        std::uint64_t next_gen(std::string const& phase)
        {
            return ++generations[phase];
        }
    };

    // -----------------------------------------------------------------
    // Handle registry + remote fetcher
    //
    // Co-rank probes need to read individual elements from input ranges
    // that may live on remote localities. Since local iterators cannot be
    // serialized, each locality registers its owned handles in a static
    // per-invocation registry. Remote localities invoke a plain HPX action
    // (fetcher) that looks up the handle and reads the requested element.
    // The RAII guard ensures cleanup after the kernel completes.
    // -----------------------------------------------------------------

    ///////////////////////////////////////////////////////////////////////////
    // Static per-locality registry for slice handles. Tagged by RegistryTag
    // so that A and B registries are always distinct, even when value types
    // and iterator types are the same.
    template <typename RegistryTag, typename T, typename LocalIter>
    struct distributed_merge_registry
    {
        using handle_type = distributed_merge_owned_handle<LocalIter>;

        static hpx::spinlock& mtx()
        {
            static hpx::spinlock m;
            return m;
        }

        static auto& map()
        {
            static std::unordered_map<std::string,
                std::unordered_map<std::uint64_t, handle_type>>
                registry;
            return registry;
        }
    };

    // RAII guard that registers handles on construction and deregisters
    // on destruction.
    template <typename RegistryTag, typename T, typename LocalIter>
    struct distributed_merge_registry_guard
    {
        using registry_type =
            distributed_merge_registry<RegistryTag, T, LocalIter>;
        using handle_type = typename registry_type::handle_type;

        distributed_merge_registry_guard(
            std::string ns, std::vector<handle_type> const& handles)
          : invocation_ns_(HPX_MOVE(ns))
        {
            std::unordered_map<std::uint64_t, handle_type> indexed;
            indexed.reserve(handles.size());
            for (auto const& h : handles)
            {
                indexed.emplace(h.slice_id, h);
            }

            std::lock_guard<hpx::spinlock> lk(registry_type::mtx());
            registry_type::map().emplace(invocation_ns_, HPX_MOVE(indexed));
        }

        ~distributed_merge_registry_guard()
        {
            std::lock_guard<hpx::spinlock> lk(registry_type::mtx());
            registry_type::map().erase(invocation_ns_);
        }

        distributed_merge_registry_guard(
            distributed_merge_registry_guard const&) = delete;
        distributed_merge_registry_guard& operator=(
            distributed_merge_registry_guard const&) = delete;

    private:
        std::string invocation_ns_;
    };

    ///////////////////////////////////////////////////////////////////////////
    // Registry tag types ensure A and B never collide.
    struct distributed_merge_tag_A
    {
    };
    struct distributed_merge_tag_B
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    // Remote value fetcher: reads a single value from a registered slice.
    // Used by co-rank probes when the target slice is on a remote locality.
    template <typename RegistryTag, typename T, typename LocalIter>
    struct distributed_merge_fetcher
    {
        using registry_type =
            distributed_merge_registry<RegistryTag, T, LocalIter>;
        using local_traits =
            hpx::traits::segmented_local_iterator_traits<LocalIter>;

        static T call(std::string const& invocation_ns, std::uint64_t slice_id,
            std::uint64_t offset)
        {
            std::lock_guard<hpx::spinlock> lk(registry_type::mtx());

            auto& reg = registry_type::map();
            auto ns_it = reg.find(invocation_ns);
            if (ns_it == reg.end())
            {
                HPX_THROW_EXCEPTION(hpx::error::bad_parameter,
                    "distributed_merge_fetcher::call",
                    "slice registry entry not found for invocation "
                    "namespace");
            }

            auto& handle_map = ns_it->second;
            auto handle_it = handle_map.find(slice_id);
            if (handle_it == handle_map.end())
            {
                HPX_THROW_EXCEPTION(hpx::error::bad_parameter,
                    "distributed_merge_fetcher::call",
                    "slice handle not found for slice_id");
            }

            auto raw = local_traits::local(handle_it->second.local_begin);
            std::advance(raw, static_cast<std::ptrdiff_t>(offset));
            return *raw;
        }

        struct action
          : hpx::actions::make_action_t<
                decltype(&distributed_merge_fetcher::call),
                &distributed_merge_fetcher::call, action>
        {
        };
    };

    ///////////////////////////////////////////////////////////////////////////
    // Helper: binary search a sorted layout to find the slice containing a
    // given global rank.
    inline std::uint64_t distributed_merge_find_slice(
        std::vector<distributed_merge_slice_metadata> const& layout,
        std::uint64_t global_rank)
    {
        auto it = std::upper_bound(layout.begin(), layout.end(), global_rank,
            [](std::uint64_t rank, distributed_merge_slice_metadata const& s) {
                return rank < s.global_end;
            });

        if (it == layout.end() || global_rank < it->global_begin)
        {
            HPX_THROW_EXCEPTION(hpx::error::bad_parameter,
                "distributed_merge_find_slice",
                "global rank outside layout bounds");
        }

        return static_cast<std::uint64_t>(std::distance(layout.begin(), it));
    }

    ///////////////////////////////////////////////////////////////////////////
    // build_ordered_slices: decompose a segmented range [first, last) into
    // globally ordered slice metadata + per-locality handle maps.
    template <typename SegIter>
    using distributed_merge_local_iter_t =
        typename hpx::traits::segmented_iterator_traits<
            SegIter>::local_iterator;

    template <typename SegIter>
    using distributed_merge_handle_t =
        distributed_merge_owned_handle<distributed_merge_local_iter_t<SegIter>>;

    template <typename SegIter>
    using distributed_merge_handle_map_t = std::map<std::uint32_t,
        std::vector<distributed_merge_handle_t<SegIter>>>;

    template <typename SegIter>
    std::pair<std::vector<distributed_merge_slice_metadata>,
        distributed_merge_handle_map_t<SegIter>>
    distributed_merge_build_slices(SegIter first, SegIter last)
    {
        using traits = hpx::traits::segmented_iterator_traits<SegIter>;
        using segment_iterator = typename traits::segment_iterator;
        using local_iterator = typename traits::local_iterator;

        std::vector<distributed_merge_slice_metadata> layout;
        distributed_merge_handle_map_t<SegIter> handles;

        if (first == last)
        {
            return {HPX_MOVE(layout), HPX_MOVE(handles)};
        }

        std::uint64_t next_id = 0;
        std::uint64_t offset = 0;

        auto add_slice = [&](segment_iterator const& seg, local_iterator lb,
                             local_iterator le) {
            auto const len = static_cast<std::uint64_t>(std::distance(lb, le));
            if (len == 0)
                return;

            auto const id = traits::get_id(seg);
            auto const loc_id = naming::get_locality_id_from_id(id);

            layout.push_back(distributed_merge_slice_metadata{
                next_id, 0, offset, offset + len, loc_id});

            handles[loc_id].push_back(
                distributed_merge_owned_handle<local_iterator>{
                    next_id, offset, offset + len, HPX_MOVE(lb)});

            ++next_id;
            offset += len;
        };

        segment_iterator sit = traits::segment(first);
        segment_iterator send = traits::segment(last);

        if (sit == send)
        {
            add_slice(sit, traits::local(first), traits::local(last));
            return {HPX_MOVE(layout), HPX_MOVE(handles)};
        }

        add_slice(sit, traits::local(first), traits::end(sit));

        for (++sit; sit != send; ++sit)
        {
            add_slice(sit, traits::begin(sit), traits::end(sit));
        }

        add_slice(send, traits::begin(send), traits::local(last));

        return {HPX_MOVE(layout), HPX_MOVE(handles)};
    }

    ///////////////////////////////////////////////////////////////////////////
    // Collect all unique participant localities from A, B, D layouts.
    inline std::vector<hpx::id_type>
    distributed_merge_collect_participants_and_assign_sites(
        std::vector<distributed_merge_slice_metadata>& A_layout,
        std::vector<distributed_merge_slice_metadata>& B_layout,
        std::vector<distributed_merge_slice_metadata>& D_layout)
    {
        // Collect unique locality IDs and assign site indices.
        // std::map keeps keys sorted, giving deterministic site ordering.
        std::map<std::uint32_t, std::uint64_t> loc_to_site;
        auto collect =
            [&loc_to_site](
                std::vector<distributed_merge_slice_metadata> const& l) {
                for (auto const& s : l)
                    loc_to_site.emplace(s.locality_id, 0);
            };
        collect(A_layout);
        collect(B_layout);
        collect(D_layout);

        // Assign contiguous site indices and build participants.
        std::vector<hpx::id_type> participants;
        participants.reserve(loc_to_site.size());
        std::uint64_t site = 0;
        for (auto& [loc_id, idx] : loc_to_site)
        {
            idx = site++;
            participants.push_back(naming::get_id_from_locality_id(loc_id));
        }

        // Assign site_index in all layouts.
        for (auto& s : A_layout)
            s.site_index = loc_to_site[s.locality_id];
        for (auto& s : B_layout)
            s.site_index = loc_to_site[s.locality_id];
        for (auto& s : D_layout)
            s.site_index = loc_to_site[s.locality_id];

        return participants;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Get handles for a specific locality, or empty vector if none.
    template <typename HandleMap>
    auto distributed_merge_get_handles(
        HandleMap const& handles, std::uint32_t locality_id)
    {
        using mapped_type = typename HandleMap::mapped_type;
        auto it = handles.find(locality_id);
        if (it == handles.end())
            return mapped_type{};
        return it->second;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Unique invocation namespace generator.
    inline std::string distributed_merge_make_ns()
    {
        static std::atomic<std::uint64_t> counter{0};
        return "/hpx/distributed_merge/" +
            std::to_string(hpx::agas::get_locality_id()) + "/" +
            std::to_string(++counter);
    }

    // -----------------------------------------------------------------
    // SPMD kernel
    //
    // One instance runs on every participating locality. Each instance
    // owns a subset of A, B, and D slices. The six phases are fully
    // symmetric. Every locality executes the same code path, differing
    // only in which slices it owns and which data it sends/receives.
    // Communication is purely collective (all_gather, all_to_all), so
    // there are no point-to-point messages to manage.
    // -----------------------------------------------------------------

    ///////////////////////////////////////////////////////////////////////////
    // The distributed merge SPMD kernel. Launched as an action on every
    // participating locality.
    template <typename Value1, typename Value2, typename LocalIter1,
        typename LocalIter2, typename LocalDestIter, typename Comp>
    struct distributed_merge_kernel
    {
        using handle1_t = distributed_merge_owned_handle<LocalIter1>;
        using handle2_t = distributed_merge_owned_handle<LocalIter2>;
        using handle_dest_t = distributed_merge_owned_handle<LocalDestIter>;
        using batch1_t = distributed_merge_payload_batch<Value1>;
        using batch2_t = distributed_merge_payload_batch<Value2>;
        using desc_t = distributed_merge_fragment_desc;
        using fetch_action1_t =
            typename distributed_merge_fetcher<distributed_merge_tag_A, Value1,
                LocalIter1>::action;
        using fetch_action2_t =
            typename distributed_merge_fetcher<distributed_merge_tag_B, Value2,
                LocalIter2>::action;

        ///////////////////////////////////////////////////////////////////////
        // Cached remote/local element fetch for co-rank probing.
        template <typename Val, typename LIter, typename FetchAction,
            typename HandleT>
        static Val fetch_at_rank(std::uint64_t rank,
            std::vector<distributed_merge_slice_metadata> const& layout,
            std::unordered_map<std::uint64_t, HandleT const*> const& idx,
            std::unordered_map<std::uint64_t, Val>& cache,
            std::uint64_t this_site,
            std::vector<hpx::id_type> const& participants,
            std::string const& invocation_ns)
        {
            auto cit = cache.find(rank);
            if (cit != cache.end())
                return cit->second;

            std::uint64_t const si = distributed_merge_find_slice(layout, rank);
            auto const& slice = layout[si];
            std::uint64_t const off = rank - slice.global_begin;

            Val val;
            if (slice.site_index == this_site)
            {
                using lt = hpx::traits::segmented_local_iterator_traits<LIter>;

                auto it = idx.find(slice.slice_id);
                HPX_ASSERT(it != idx.end());
                auto raw = lt::local(it->second->local_begin);
                std::advance(raw, static_cast<std::ptrdiff_t>(off));
                val = *raw;
            }
            else
            {
                val = hpx::async(FetchAction{}, participants[slice.site_index],
                    invocation_ns, slice.slice_id, off)
                          .get();
            }

            cache.emplace(rank, val);
            return val;
        }

        ///////////////////////////////////////////////////////////////////////
        // Stable co-rank: find (i, j) with i + j = K such that the first K
        // merged elements are exactly A[0,i) and B[0,j), with A-before-B
        // on equal keys. [lo, hi] optionally narrow the search range on
        // the A-index (clamped to the feasible interval).
        template <typename AAt, typename BAt>
        static std::pair<std::uint64_t, std::uint64_t> co_rank(std::uint64_t K,
            std::uint64_t n1, std::uint64_t n2, AAt&& A_at, BAt&& B_at,
            Comp const& comp, std::uint64_t lo = 0,
            std::uint64_t hi = (std::numeric_limits<std::uint64_t>::max)())
        {
            std::uint64_t const feasible_lo = (K > n2) ? (K - n2) : 0;
            std::uint64_t const feasible_hi = (std::min) (K, n1);

            lo = (std::max) (lo, feasible_lo);
            hi = (std::min) (hi, feasible_hi);

            while (true)
            {
                std::uint64_t const i = lo + (hi - lo) / 2;
                std::uint64_t const j = K - i;

                if (i > 0 && j < n2 && comp(B_at(j), A_at(i - 1)))
                {
                    hi = i - 1;
                }
                else if (j > 0 && i < n1 && !comp(B_at(j - 1), A_at(i)))
                {
                    lo = i + 1;
                }
                else
                {
                    return {i, j};
                }
            }
        }

        ///////////////////////////////////////////////////////////////////////
        // Local stable merge of reconstructed A and B fragments into a
        // destination slice.
        // Resolved fragment: a pointer+count span into a received batch's
        // values buffer, with global rank for ordering.
        template <typename T>
        struct resolved_fragment
        {
            std::uint64_t input_global_begin;
            T const* data;
            std::uint64_t count;
        };

        template <typename TA, typename TB>
        static void merge_into_dest_impl(handle_dest_t const& dest_handle,
            std::vector<resolved_fragment<TA>>& A_frags,
            std::vector<resolved_fragment<TB>>& B_frags, Comp const& comp)
        {
            using dest_local_traits =
                hpx::traits::segmented_local_iterator_traits<LocalDestIter>;

            auto by_rank = [](auto const& a, auto const& b) {
                return a.input_global_begin < b.input_global_begin;
            };
            std::sort(A_frags.begin(), A_frags.end(), by_rank);
            std::sort(B_frags.begin(), B_frags.end(), by_rank);

            std::uint64_t a_fi = 0, a_off = 0;
            std::uint64_t b_fi = 0, b_off = 0;

            auto a_done = [&]() { return a_fi >= A_frags.size(); };
            auto b_done = [&]() { return b_fi >= B_frags.size(); };
            auto a_val = [&]() -> TA const& {
                return A_frags[a_fi].data[a_off];
            };
            auto b_val = [&]() -> TB const& {
                return B_frags[b_fi].data[b_off];
            };
            auto a_advance = [&]() {
                if (++a_off >= A_frags[a_fi].count)
                {
                    ++a_fi;
                    a_off = 0;
                }
            };
            auto b_advance = [&]() {
                if (++b_off >= B_frags[b_fi].count)
                {
                    ++b_fi;
                    b_off = 0;
                }
            };

            auto dest_raw = dest_local_traits::local(dest_handle.local_begin);

            while (!a_done() || !b_done())
            {
                if (a_done())
                {
                    *dest_raw = b_val();
                    b_advance();
                }
                else if (b_done())
                {
                    *dest_raw = a_val();
                    a_advance();
                }
                else if (comp(b_val(), a_val()))
                {
                    *dest_raw = b_val();
                    b_advance();
                }
                else
                {
                    *dest_raw = a_val();
                    a_advance();
                }
                ++dest_raw;
            }
        }

        ///////////////////////////////////////////////////////////////////////
        // Pack fragments from local handles into per-peer batches.
        template <typename Val, typename LIter, typename HandleT,
            typename BeginRankFn, typename EndRankFn>
        static void pack_batches(std::vector<HandleT> const& handles,
            std::vector<distributed_merge_interval> const& intervals,
            BeginRankFn begin_rank_of, EndRankFn end_rank_of,
            std::vector<distributed_merge_payload_batch<Val>>& send)
        {
            using lt = hpx::traits::segmented_local_iterator_traits<LIter>;

            for (auto const& h : handles)
            {
                for (auto const& iv : intervals)
                {
                    std::uint64_t const ob =
                        (std::max) (h.global_begin, begin_rank_of(iv));
                    std::uint64_t const oe =
                        (std::min) (h.global_end, end_rank_of(iv));

                    if (ob >= oe)
                        continue;

                    auto& batch = send[iv.site_index];
                    std::uint64_t const off = batch.values.size();
                    std::uint64_t const cnt = oe - ob;

                    batch.fragments.push_back(distributed_merge_fragment_desc{
                        iv.dest_slice_id, ob, off, cnt});

                    auto raw = lt::local(h.local_begin);
                    std::advance(
                        raw, static_cast<std::ptrdiff_t>(ob - h.global_begin));
                    for (std::uint64_t k = 0; k != cnt; ++k, ++raw)
                        batch.values.push_back(*raw);
                }
            }
        }

        ///////////////////////////////////////////////////////////////////////
        // The main kernel entry point, executed as an action on each
        // participating locality.
        static void call(std::string invocation_ns,
            std::vector<hpx::id_type> participants, std::uint64_t this_site,
            std::vector<distributed_merge_slice_metadata> A_layout,
            std::vector<distributed_merge_slice_metadata> B_layout,
            std::vector<distributed_merge_slice_metadata> D_layout,
            std::vector<handle1_t> A_handles, std::vector<handle2_t> B_handles,
            std::vector<handle_dest_t> D_handles, std::uint64_t n1,
            std::uint64_t n2, Comp comp)
        {
            // --- Register local slice handles in static registries ---
            distributed_merge_registry_guard<distributed_merge_tag_A, Value1,
                LocalIter1>
                guard_A(invocation_ns, A_handles);
            distributed_merge_registry_guard<distributed_merge_tag_B, Value2,
                LocalIter2>
                guard_B(invocation_ns, B_handles);

            std::uint64_t const P = participants.size();
            distributed_merge_collective_ctx ctx(HPX_MOVE(invocation_ns));

            // --- Phase 1: Startup barrier ---
            // Ensures all localities have registered their handles before
            // any remote co-rank probes can fire.
            {
                auto bn = ctx.basename("startup_sync");
                [[maybe_unused]] auto sync_result =
                    hpx::collectives::all_gather(hpx::launch::sync, bn.c_str(),
                        std::uint8_t(1), hpx::collectives::num_sites_arg(P),
                        hpx::collectives::this_site_arg(this_site),
                        hpx::collectives::generation_arg(
                            ctx.next_gen("startup_sync")),
                        hpx::collectives::root_site_arg(0));
            }

            // --- Build indexed handle maps for O(1) lookup ---
            std::unordered_map<std::uint64_t, handle1_t const*> A_idx;
            for (auto const& h : A_handles)
                A_idx.emplace(h.slice_id, &h);

            std::unordered_map<std::uint64_t, handle2_t const*> B_idx;
            for (auto const& h : B_handles)
                B_idx.emplace(h.slice_id, &h);

            // --- Cached co-rank probe functions ---
            std::unordered_map<std::uint64_t, Value1> A_cache;
            std::unordered_map<std::uint64_t, Value2> B_cache;

            auto A_at = [&](std::uint64_t rank) -> Value1 {
                return fetch_at_rank<Value1, LocalIter1, fetch_action1_t>(rank,
                    A_layout, A_idx, A_cache, this_site, participants,
                    ctx.invocation_ns);
            };
            auto B_at = [&](std::uint64_t rank) -> Value2 {
                return fetch_at_rank<Value2, LocalIter2, fetch_action2_t>(rank,
                    B_layout, B_idx, B_cache, this_site, participants,
                    ctx.invocation_ns);
            };

            // --- Phase 2: Co-rank planning per owned dest slice ---
            // Successive slices narrow the search range: the end
            // boundary of slice k bounds the start of slice k+1.
            std::vector<distributed_merge_interval> local_intervals;
            local_intervals.reserve(D_handles.size());

            std::uint64_t prev_i = 0, prev_j = 0;

            for (auto const& dh : D_handles)
            {
                auto dit = std::find_if(D_layout.begin(), D_layout.end(),
                    [&dh](distributed_merge_slice_metadata const& s) {
                        return s.slice_id == dh.slice_id;
                    });
                HPX_ASSERT(dit != D_layout.end());

                auto const [i0, j0] = co_rank(dit->global_begin, n1, n2, A_at,
                    B_at, comp, prev_i, dit->global_begin - prev_j);
                auto const [i1, j1] = co_rank(dit->global_end, n1, n2, A_at,
                    B_at, comp, i0, dit->global_end - j0);

                local_intervals.push_back(distributed_merge_interval{
                    dit->slice_id, dit->site_index, i0, i1, j0, j1});

                prev_i = i1;
                prev_j = j1;
            }

            // --- Phase 3: Share intervals via all_gather ---
            auto intervals_bn = ctx.basename("intervals");
            auto gathered = hpx::collectives::all_gather(hpx::launch::sync,
                intervals_bn.c_str(), HPX_MOVE(local_intervals),
                hpx::collectives::num_sites_arg(P),
                hpx::collectives::this_site_arg(this_site),
                hpx::collectives::generation_arg(ctx.next_gen("intervals")),
                hpx::collectives::root_site_arg(0));

            // Flatten gathered intervals
            std::vector<distributed_merge_interval> all_intervals;
            {
                std::uint64_t total = 0;
                for (auto const& v : gathered)
                    total += v.size();
                all_intervals.reserve(total);
                for (auto& v : gathered)
                {
                    all_intervals.insert(all_intervals.end(),
                        std::make_move_iterator(v.begin()),
                        std::make_move_iterator(v.end()));
                }
            }

            // --- Phase 4: Pack payload batches ---
            std::vector<batch1_t> send_A(P);
            std::vector<batch2_t> send_B(P);

            pack_batches<Value1, LocalIter1>(
                A_handles, all_intervals,
                [](auto const& iv) { return iv.A_begin_rank; },
                [](auto const& iv) { return iv.A_end_rank; }, send_A);
            pack_batches<Value2, LocalIter2>(
                B_handles, all_intervals,
                [](auto const& iv) { return iv.B_begin_rank; },
                [](auto const& iv) { return iv.B_end_rank; }, send_B);

            // --- Phase 5: Payload exchange ---
            auto bn_A = ctx.basename("payload_A");
            auto bn_B = ctx.basename("payload_B");

            auto recv_A =
                hpx::collectives::all_to_all(hpx::launch::sync, bn_A.c_str(),
                    HPX_MOVE(send_A), hpx::collectives::num_sites_arg(P),
                    hpx::collectives::this_site_arg(this_site),
                    hpx::collectives::generation_arg(ctx.next_gen("payload_A")),
                    hpx::collectives::root_site_arg(0));

            auto recv_B =
                hpx::collectives::all_to_all(hpx::launch::sync, bn_B.c_str(),
                    HPX_MOVE(send_B), hpx::collectives::num_sites_arg(P),
                    hpx::collectives::this_site_arg(this_site),
                    hpx::collectives::generation_arg(ctx.next_gen("payload_B")),
                    hpx::collectives::root_site_arg(0));

            // --- Phase 6: Resolve fragments and merge ---
            // Each received batch contains a flat values buffer and
            // lightweight descriptors. We resolve each descriptor to a
            // pointer directly into the buffer (zero-copy), grouped by
            // destination slice, then perform a local stable merge.
            using rfrag1_t = resolved_fragment<Value1>;
            using rfrag2_t = resolved_fragment<Value2>;

            std::unordered_map<std::uint64_t, std::vector<rfrag1_t>> A_by_dest;
            std::unordered_map<std::uint64_t, std::vector<rfrag2_t>> B_by_dest;

            for (auto& batch : recv_A)
            {
                for (auto const& f : batch.fragments)
                {
                    A_by_dest[f.dest_slice_id].push_back(rfrag1_t{
                        f.input_global_begin,
                        batch.values.data() + f.value_offset, f.value_count});
                }
            }
            for (auto& batch : recv_B)
            {
                for (auto const& f : batch.fragments)
                {
                    B_by_dest[f.dest_slice_id].push_back(rfrag2_t{
                        f.input_global_begin,
                        batch.values.data() + f.value_offset, f.value_count});
                }
            }

            // Merge into each owned destination slice
            for (auto const& dh : D_handles)
            {
                merge_into_dest_impl(
                    dh, A_by_dest[dh.slice_id], B_by_dest[dh.slice_id], comp);
            }
        }

        struct action
          : hpx::actions::make_action_t<
                decltype(&distributed_merge_kernel::call),
                &distributed_merge_kernel::call, action>
        {
        };
    };

    // -----------------------------------------------------------------
    // Coordinator
    //
    // Runs on the calling locality. Decomposes the three segmented
    // ranges into slice metadata + handle maps, determines which
    // localities participate, then launches the SPMD kernel on each
    // one via async actions. Waits for all kernels to complete and
    // propagates any errors.
    // -----------------------------------------------------------------

    ///////////////////////////////////////////////////////////////////////////
    // Coordinator: the main entry point for distributed segmented merge.
    // Called from the tag_invoke overloads.
    template <typename ExPolicy, typename SegIter1, typename SegIter2,
        typename DestIter, typename Comp>
    hpx::parallel::util::detail::algorithm_result_t<ExPolicy, DestIter>
    segmented_merge(ExPolicy&& /* policy */, SegIter1 first1, SegIter1 last1,
        SegIter2 first2, SegIter2 last2, DestIter dest, Comp&& comp)
    {
        using traits1 = hpx::traits::segmented_iterator_traits<SegIter1>;
        using traits2 = hpx::traits::segmented_iterator_traits<SegIter2>;
        using traits3 = hpx::traits::segmented_iterator_traits<DestIter>;

        using local_iterator1 = typename traits1::local_iterator;
        using local_iterator2 = typename traits2::local_iterator;
        using local_iterator3 = typename traits3::local_iterator;

        using value_type1 = typename std::iterator_traits<SegIter1>::value_type;
        using value_type2 = typename std::iterator_traits<SegIter2>::value_type;
        using comp_type = std::decay_t<Comp>;

        using result_type =
            hpx::parallel::util::detail::algorithm_result<ExPolicy, DestIter>;

        // Validate local raw iterators have random-access capability
        using local_raw1 =
            typename hpx::traits::segmented_local_iterator_traits<
                local_iterator1>::local_raw_iterator;
        using local_raw2 =
            typename hpx::traits::segmented_local_iterator_traits<
                local_iterator2>::local_raw_iterator;
        using local_raw3 =
            typename hpx::traits::segmented_local_iterator_traits<
                local_iterator3>::local_raw_iterator;

        static_assert(
            std::is_base_of_v<std::random_access_iterator_tag,
                typename std::iterator_traits<local_raw1>::iterator_category>,
            "distributed merge requires random-access local iterators "
            "for the first input");
        static_assert(
            std::is_base_of_v<std::random_access_iterator_tag,
                typename std::iterator_traits<local_raw2>::iterator_category>,
            "distributed merge requires random-access local iterators "
            "for the second input");
        static_assert(
            std::is_base_of_v<std::random_access_iterator_tag,
                typename std::iterator_traits<local_raw3>::iterator_category>,
            "distributed merge requires random-access local iterators "
            "for the destination");

        // Both empty, nothing to do.
        if (first1 == last1 && first2 == last2)
        {
            return result_type::get(HPX_MOVE(dest));
        }

        // Decompose ranges into ordered slices
        auto [A_layout, A_handles] =
            distributed_merge_build_slices(first1, last1);
        auto [B_layout, B_handles] =
            distributed_merge_build_slices(first2, last2);

        auto const n1 =
            static_cast<std::uint64_t>(std::distance(first1, last1));
        auto const n2 =
            static_cast<std::uint64_t>(std::distance(first2, last2));
        auto const N = n1 + n2;

        auto dest_end = dest;
        std::advance(dest_end, static_cast<std::ptrdiff_t>(N));

        auto [D_layout, D_handles] =
            distributed_merge_build_slices(dest, dest_end);

        if (D_layout.empty())
        {
            return result_type::get(HPX_MOVE(dest_end));
        }

        // Determine participants and assign site indices
        auto participants =
            distributed_merge_collect_participants_and_assign_sites(
                A_layout, B_layout, D_layout);

        // Create kernel action type and invocation namespace
        using kernel_type = distributed_merge_kernel<value_type1, value_type2,
            local_iterator1, local_iterator2, local_iterator3, comp_type>;
        using action_type = typename kernel_type::action;

        std::string const ns = distributed_merge_make_ns();
        comp_type const comp_copy = HPX_FORWARD(Comp, comp);

        // Launch kernel on every participant
        std::vector<hpx::future<void>> futures;
        futures.reserve(participants.size());

        for (std::uint64_t site = 0; site != participants.size(); ++site)
        {
            auto const loc_id =
                naming::get_locality_id_from_id(participants[site]);

            futures.push_back(hpx::async(action_type{}, participants[site], ns,
                participants, site, A_layout, B_layout, D_layout,
                distributed_merge_get_handles(A_handles, loc_id),
                distributed_merge_get_handles(B_handles, loc_id),
                distributed_merge_get_handles(D_handles, loc_id), n1, n2,
                comp_copy));
        }

        // Collect errors from all participants
        auto collect_errors = [](std::vector<hpx::future<void>> fs) {
            std::list<std::exception_ptr> errors;
            for (auto& f : fs)
            {
                try
                {
                    f.get();
                }
                catch (...)
                {
                    errors.push_back(std::current_exception());
                }
            }
            if (!errors.empty())
            {
                throw hpx::exception_list(HPX_MOVE(errors));
            }
        };

        // Return based on execution policy
        if constexpr (hpx::is_async_execution_policy_v<std::decay_t<ExPolicy>>)
        {
            return result_type::get(hpx::when_all(HPX_MOVE(futures))
                    .then([dest_end = HPX_MOVE(dest_end),
                              collect_errors = HPX_MOVE(collect_errors)](
                              hpx::future<std::vector<hpx::future<void>>>&&
                                  f) mutable -> DestIter {
                        collect_errors(f.get());
                        return dest_end;
                    }));
        }
        else
        {
            collect_errors(hpx::when_all(HPX_MOVE(futures)).get());
            return result_type::get(HPX_MOVE(dest_end));
        }
    }

}    // namespace hpx::parallel::detail

HPX_IS_BITWISE_SERIALIZABLE(
    hpx::parallel::detail::distributed_merge_slice_metadata)
HPX_IS_BITWISE_SERIALIZABLE(hpx::parallel::detail::distributed_merge_interval)
HPX_IS_BITWISE_SERIALIZABLE(
    hpx::parallel::detail::distributed_merge_fragment_desc)
