# Hierarchical `all_to_all` for HPX — design note

**Author:** Anshuman Agrawal
**Status:** v3, post-discussion (#7200), draft reference for prerequisite PRs
**Scope:** hierarchical `all_to_all` for the existing `hpx::collectives::hierarchical_communicator` infrastructure
**Related work:** PR #7160 (hierarchical `all_reduce` / `all_gather`, merged), PR #7193 (flat-fallback factory threshold, in review), Discussion #7200 (architectural Q&A with `@hkaiser`).

## 1. Purpose

This note records the decisions reached in #7200 and the v1 implementation shape that follows from them. It is intended as the engineering reference for the upcoming PR sequence:

1. (prerequisite) move existing two-phase hierarchical collectives from internal generation stride 2 to stride 3, and consolidate the top-level partitioning rule into a shared `detail::` helper;
2. (this design) hierarchical `all_to_all` v1 with three-phase decomposition, padded blocks, and flat fallback.

Decisions from #7200, recorded here as the current v1 direction:

- **Phase 2 communicator** — reuse the top-of-tree communicator (`communicators.get(0)` on top reps). Considered alternatives (a dedicated inter-group communicator at factory time; explicit `channel_communicator` routing) are deferred as future work, not v1.
- **Phase 2 block layout** — v1 uses padded fixed-size `g_max × g_max` blocks. Ragged blocks and `all_to_allv` over `channel_communicator` are deferred.
- **Generation namespacing** — uniform internal stride of three for hierarchical collectives sharing one `hierarchical_communicator`. Separate communicator families per collective were considered and rejected (would prevent shared communicator identity).
- **Stride-three mechanism** — no internal communicator API change, no dummy round-trips. The existing `next_generation` accepts `new_generation >= generation_` and post-increments, so two-phase collectives encode the implicit skip by issuing the last phase at the stride-3 last slot directly. Detailed in §9.
- **Large-payload fallback threshold** — not in v1. Document the per-rep memory budget; users size `arity` accordingly.

§15 lists what is explicitly out of scope.

## 2. Existing HPX behavior this design depends on

Every claim below is verified against `master` at the time of writing; line numbers are given so reviewers can sanity-check.

### 2.1 Communicator path stored per site

`hierarchical_communicator` (defined in `libs/full/collectives/include/hpx/collectives/create_communicator.hpp:290`) stores

```cpp
std::vector<hpx::tuple<communicator, this_site_arg>> communicators;
```

ordered top-to-bottom: `communicators[0]` is the highest level the *current site* participates in; `communicators[size-1]` is its terminal flat (leaf) communicator.

The recursion that fills the vector (`recursively_fill_communicators`, `libs/full/collectives/src/create_communicator.cpp:375`) only adds an entry at a given level if `this_site == current_left` for that level's group — i.e., the current site is the leftmost site of the group, which is the group's representative. Therefore:

- For the leftmost site of a top-level group (a top rep), `communicators[0]` is the **top-of-tree communicator** connecting all top reps.
- For any non-top-rep, `communicators[0]` is the highest **subtree-internal** communicator the site participates in, never the top-of-tree.

The implementation must not treat `communicators.get(0)` as "the top communicator" on every site.

### 2.2 Partitioning rule

At each non-terminal level the range `[left, right]` is split into `arity` groups using

```cpp
std::size_t division_steps = (right - left + 1) / arity;
std::size_t remainder      = (right - left + 1) % arity;
std::size_t current_group_size = division_steps + (i < remainder ? 1 : 0);
```

(`recursively_fill_communicators`, lines 396–403). The first `remainder` groups are larger by one. The recursion's terminal condition is

```cpp
if (right - left < arity) { /* create a single flat communicator and stop */ }
```

so a range of `arity` or fewer sites becomes a single terminal flat communicator. Within a non-terminal group, the rank of a representative inside its parent's `arity`-wide group communicator is its index `i` in the partitioning loop above.

### 2.3 Walks in existing hierarchical gather and scatter

#### Gather, root path (`gather.hpp:575`):

```cpp
std::vector<T> result(1, local_result);
for (std::size_t i = communicators.size() - 1; i != 0; --i) {
    result = detail::gather_data(
        gather_here(hpx::launch::sync, communicators.get(i),
            HPX_MOVE(result), this_site_arg(0), generation));
}
return detail::gather_data(
    gather_here(communicators.get(0), HPX_MOVE(result),
        this_site_arg(0), generation));
```

The loop walks subtree-internal levels bottom-up; the **final call after the loop is on `communicators.get(0)`**. On a top rep this final call is *across the top of the tree*. Phase 1 of `all_to_all` needs the walk to stop *before* this final call on top reps. This is the helper requirement in §6.

#### Gather, non-root path (`gather.hpp:699`):

The non-root variant does the same loop and then `gather_there(communicators.get(0), …)`. On a non-top-rep, `communicators.get(0)` is the highest *subtree-internal* communicator, so this final call already sends within the subtree to the next-level-up representative. Phase 1 needs no change for non-top-reps.

#### Scatter, root path (`scatter.hpp:702`):

```cpp
arity_arg arity = communicators.get_arity();
std::vector<T> data = scatter_to(hpx::launch::sync,
    communicators.get(0),
    detail::scatter_data(HPX_MOVE(local_result), arity),
    this_site_arg(0), generation);

for (std::size_t i = 1; i < communicators.size() - 1; ++i) {
    data = scatter_to(hpx::launch::sync, communicators.get(i),
        detail::scatter_data(HPX_MOVE(data), arity),
        this_site_arg(this_site % arity), generation);
}

return scatter_to(communicators.back(),
    HPX_MOVE(data), this_site_arg(0), generation);
```

The **first call before the loop is on `communicators.get(0)`**. On a top rep this is the cross-top scatter. Phase 3 needs to start at `communicators.get(1)` instead — the helper requirement in §8.

Note on the loop body's `this_site_arg(this_site % arity)`: a top rep is the leftmost site of its group at every subtree-internal level it appears on, so within its own subtree `this_site % arity` evaluates to `0` at each level. The Phase 3 helper can use `this_site_arg(0)` uniformly inside the subtree.

#### Scatter, non-root path (`scatter.hpp:537`):

The non-root variant starts with `scatter_from(communicators.get(0), …)`. On a non-top-rep, `communicators.get(0)` is subtree-internal, so this initial call already receives within the subtree from the next-level-up representative. Phase 3 needs no change for non-top-reps.

### 2.4 Flat fallback (PR #7193)

When the configured threshold causes the factory to emit a flat communicator instead of a hierarchical one, the resulting `hierarchical_communicator` has `communicators.size() == 1`. Existing hierarchical `scatter_to` and `scatter_from` short-circuit on this condition (`scatter.hpp:547`, `scatter.hpp:713`). Hierarchical `all_to_all` mirrors this exactly (§10).

`flat_fallback_threshold_arg` is defined in `argument_types.hpp:91-92` with a default value of 16.

### 2.5 `next_generation` semantics

The internal gate's `next_generation`, used at `libs/full/collectives/include/hpx/collectives/detail/communicator.hpp:447`, accepts any `new_generation >= generation_` and **post-increments** the gate's internal counter. This is the foundation for stride-three handling (§9) and means no internal communicator API change is required for this design.

### 2.6 Existing two-phase generation arithmetic

`all_reduce.hpp:459-460` and `all_gather.hpp:420-421` currently use

```cpp
generation_arg const reduce_gen   (2 * generation - 1);
generation_arg const broadcast_gen(2 * generation);
```

The `bad_parameter` strings on those paths describe this as "the 2k/2k+1 internal mapping" (`all_reduce.hpp:451`, `all_gather.hpp:412`) — which is a pre-existing documentation slip; the actual arithmetic is `2k-1, 2k`. The prerequisite PR (§14, step 0) updates both the arithmetic to `3k-2, 3k` and the strings.

### 2.7 Flat `all_to_all` validation

Flat `all_to_all` requires every contribution to be a `vector<T>` of length `num_sites`; mismatch is a `bad_parameter` exceptional future thrown from the per-site finalizer (`all_to_all.hpp:268`). It also rejects `generation == 0` (`all_to_all.hpp:297`). Hierarchical `all_to_all` mirrors both behaviors at the entry point.

## 3. Problem statement

For `N` sites, each site `i` contributes `In[i]` of length `N`, where `In[i][j]` is the value intended for site `j`. After the collective, site `i` receives

```text
Out[i][j] = In[j][i]
```

— the transpose of the logical `N × N` matrix.

Unlike `all_reduce` and `all_gather`, `all_to_all` cannot be expressed as "compute at one root, then broadcast": every site's output is different. A hierarchical implementation cannot reduce the *total* logical data volume, but it can reduce the *number of inter-node messages* by aggregating within top-level subtrees, exchanging between representatives, and redistributing within destination subtrees.

The v1 algorithm is the standard three-phase hierarchical decomposition:

1. intra-subtree gather to the top representative (Phase 1);
2. inter-representative `all_to_all` over the top communicator (Phase 2);
3. intra-subtree scatter from the top representative (Phase 3).

## 4. API

```cpp
template <typename T>
hpx::future<std::vector<T>> all_to_all(
    hierarchical_communicator const& communicators,
    std::vector<T>&& local_result,
    this_site_arg this_site = this_site_arg(),
    generation_arg generation = generation_arg());

template <typename T>
hpx::future<std::vector<T>> all_to_all(
    hierarchical_communicator const& communicators,
    std::vector<T>&& local_result,
    generation_arg generation,
    this_site_arg this_site = this_site_arg());
```

Synchronous overloads use the existing `hpx::launch::sync_policy` pattern, mirroring `all_reduce.hpp:485-494` and `all_gather.hpp:444-454`.

**Validation at entry:**

- If `generation` is the default (`is_default()` true), return a `bad_parameter` exceptional future stating that hierarchical `all_to_all` requires an explicit generation number for the `3k-2/3k-1/3k` internal mapping. This mirrors the existing all_reduce/all_gather validation pattern (`all_reduce.hpp:445-452`).
- If `local_result.size() != num_sites`, return a `bad_parameter` exceptional future with the same message style as flat `all_to_all`'s finalizer (`all_to_all.hpp:268`).

## 5. Top-level group helper

Phase 1, Phase 2 padding/reconstruction, and Phase 3 all need to know which top-level group a given site belongs to and the boundaries of every other top-level group. The information is recoverable from `(num_sites, arity, global_site)` plus the partitioning rule in §2.2.

The current state of the codebase encodes that rule once, inside `recursively_fill_communicators`. Duplicating the rule into a separate helper risks silent drift between the factory and the helper. The right move is to lift the top-level split into a shared `detail::` function used by both call sites:

```cpp
namespace hpx::collectives::detail {

struct top_level_group {
    std::size_t index;          // group id in [0, groups.size())
    std::size_t left;           // global first site
    std::size_t right;          // global last site
    std::size_t size;           // right - left + 1
};

// Canonical top-level partitioning of [0, num_sites). Returned vector has
// size equal to the number of top-level groups, indexed by group id.
// Used by both recursively_fill_communicators (top frame) and the
// all_to_all helpers.
std::vector<top_level_group> get_top_level_groups(
    std::size_t num_sites, std::size_t arity);

// Convenience: which top-level group does this site belong to, and is it
// that group's representative? Implemented as a linear walk over
// get_top_level_groups (arity is small).
struct site_group_info {
    std::size_t group_index;
    std::size_t local_index;     // global_site - groups[group_index].left
    bool is_representative;      // global_site == groups[group_index].left
};

site_group_info classify_site(
    std::vector<top_level_group> const& groups, std::size_t global_site);

} // namespace hpx::collectives::detail
```

The two helpers separate concerns cleanly: `get_top_level_groups` is the iteration primitive (used by the factory's top frame and by Phase 2 reconstruction in §7.3); `classify_site` is the lookup primitive (used to decide whether the current site is a top rep in the all_to_all entry point).

When `num_sites <= arity`, the factory reaches the terminal flat case immediately (§2.2) and the resulting `hierarchical_communicator` has `size() == 1`; the all_to_all entry point delegates to flat `all_to_all` *before* using these helpers (§10), so the helpers do not need a `size() == 1` branch. `get_top_level_groups` is also defined for `num_sites <= arity` (it returns one group per site, each of size 1) for use in tests; production all_to_all paths never reach it in that regime.

The prerequisite PR (§14, step 1) ports `recursively_fill_communicators`'s top-frame split to consume `get_top_level_groups`, ensuring both call sites stay in lockstep by construction. A regression test exercises the helper against a known partition table for several `(N, arity)` pairs, including the unbalanced cases listed in §12.6.

## 6. Phase 1 — intra-subtree gather

After Phase 1, the representative `rep(k) = left(G_k)` of top-level group `G_k` holds

```text
Row_k = [ In[member_0(G_k)], In[member_1(G_k)], ..., In[member_{|G_k|-1}(G_k)] ]
```

— a `vector<vector<T>>` with `|G_k|` rows and `N` columns.

**Non-representatives.** The existing public hierarchical `gather_there` is correct *as is* for Phase 1, because `communicators.get(0)` on a non-top-rep is already the highest subtree-internal communicator (§2.3). The site simply calls `gather_there(...)` with the Phase 1 generation.

**Top representatives.** A top rep's existing public hierarchical `gather_here` would, after the loop, call `gather_here(communicators.get(0), …)` on the top communicator and gather across reps (§2.3). That last call must be skipped:

```cpp
template <typename T>
hpx::future<std::vector<std::vector<T>>>
subtree_gather_at_top_rep(
    hierarchical_communicator const& communicators,
    std::vector<T>&& local_result,
    generation_arg generation);
```

Implementation is `gather_here`'s body minus the final cross-top call:

```cpp
std::vector<std::vector<T>> result(1, HPX_MOVE(local_result));
for (std::size_t i = communicators.size() - 1; i != 0; --i)
{
    result = detail::gather_data(
        gather_here(hpx::launch::sync, communicators.get(i),
            HPX_MOVE(result), this_site_arg(0), generation));
}
return hpx::make_ready_future(HPX_MOVE(result));
```

Edge case: a top rep whose top-level group has size 1 (possible when `num_sites % arity != 0`, e.g., `N=5, arity=4` for sites 2, 3, 4) still has the top communicator *plus* a terminal single-site communicator in its stored path; `communicators.size() == 2`. The loop runs once, performing a degenerate single-participant gather on the terminal communicator, which returns the input unchanged. The helper does not need a special case for this. The only situation where `communicators.size() == 1` is the flat-fallback / `num_sites <= arity` path, which is short-circuited at the entry point (§10) before any helper is called.

The Phase 1 internal generation is `generation_arg(3 * generation - 2)`; the same value is used by the helper above and by the public `gather_there` calls on non-top-reps.

## 7. Phase 2 — inter-representative `all_to_all`

Only top reps enter Phase 2. On those sites `communicators.get(0)` is the top communicator.

Each rep `rep(k)` has `Row_k` from Phase 1. It must produce, for each destination top group `dst`, an outgoing block

```text
block[k → dst] = Row_k restricted to columns belonging to G_dst
              = |G_k| × |G_dst| sub-matrix
```

and exchange these via flat `all_to_all` over `communicators.get(0)`.

### 7.1 Why padding (with the actual contract)

Flat `all_to_all<vector<T>>` does **not** require the inner vectors to be the same length — they are serialized payloads, so each `vector<T>` element of the contribution can carry a different number of `T`s. `@hkaiser` confirmed this in #7200: *"Serializing a vector<vector<T>> should be no problem as long as T is serializable."*

Padding is therefore a **v1 simplicity choice for receiver-side reconstruction**, not a serialization requirement. The doc must not claim otherwise. The cost is bounded: top-group sizes differ by at most one (§2.2), so for a max group size `g_max` the padding waste per block is at most `2 g_max - 1` slots out of `g_max²`.

**Default-constructibility caveat.** Materializing padding slots requires a pad value, which adds a `std::default_initializable<T>` requirement on the hierarchical path that flat `all_to_all` does not impose. This is a real narrowing of the type contract. The padding direction from #7200 implies this as a documented v1 limitation unless reviewers prefer moving directly to ragged blocks; the ragged-block follow-up (§15) lifts it. The receiver-side reconstruction logic in §7.3 is structurally identical for ragged, with `g_max` replaced by per-block `(src.size, my.size)` in the index calculation, so the choice between "ship padding now, lift with ragged" and "skip the intermediate state" is one reviewers can revisit without redesigning Phase 2.

### 7.2 Padded representation

Let `g_max = max_k |G_k|`. Every block `block[k → dst]` is materialized as a flat `vector<T>` of length `g_max²`, laid out row-major:

```text
block[k → dst][r * g_max + c]
    = Row_k[r][ left(G_dst) + c ]   if r < |G_k|  and  c < |G_dst|
    = T{}                           otherwise (padding; ignored on receive)
```

The Phase 2 call from a top rep:

```cpp
generation_arg const phase2_gen(3 * generation - 1);

auto const groups = detail::get_top_level_groups(num_sites, arity);
std::size_t const num_top_groups = groups.size();

std::vector<std::vector<T>> outgoing(num_top_groups);
for (std::size_t dst = 0; dst != num_top_groups; ++dst) {
    outgoing[dst] = build_padded_block(Row_k, /*src=*/k, dst, g_max);
}

std::vector<std::vector<T>> received =
    all_to_all<std::vector<T>>(communicators.get(0),
        HPX_MOVE(outgoing), communicators.site(0), phase2_gen).get();
```

`outgoing.size() == received.size() == num_top_groups` always; that is the flat `all_to_all` contract on `communicators.get(0)`, whose `num_sites` equals `num_top_groups`.

### 7.3 Reconstruction (receiver needs the *source* group's size)

`received[src]` is `block[src → k]`, padded to `g_max²`. To extract the real region the receiver needs `|G_src|` and `|G_k|`. Both come directly from `get_top_level_groups`:

```cpp
auto const groups = detail::get_top_level_groups(num_sites, arity);
auto const my_info = detail::classify_site(groups, this_site);
auto const& my_group = groups[my_info.group_index];
// my_group.size == |G_k|

std::vector<std::vector<T>> Col_k(my_group.size, std::vector<T>(num_sites));

for (std::size_t src = 0; src != groups.size(); ++src) {
    auto const& src_group = groups[src];
    auto const& blk = received[src];
    for (std::size_t r = 0; r != src_group.size; ++r) {
        for (std::size_t c = 0; c != my_group.size; ++c) {
            std::size_t const global_src_member = src_group.left + r;
            Col_k[c][global_src_member] = blk[r * g_max + c];
            // blk slots with r >= src_group.size or c >= my_group.size are
            // padding and are never read.
        }
    }
}
```

The contract from §3 holds:

```
Col_k[c][global_src_member]
  = blk[r * g_max + c]
  = Row_src[r][ left(G_k) + c ]                  (Phase 2 sender's layout)
  = In[src_group.left + r][ my_group.left + c ]
  = In[global_src_member][ member_c(G_k) ]
  = Out[member_c(G_k)][global_src_member]        (Phase 3 contract)
```

so `Col_k[c]` is exactly the output vector that member `c` of `G_k` will receive at the end of Phase 3.

### 7.4 Worked example: `N = 11, arity = 4`

Top groups, from §2.2: `[0..2], [3..5], [6..8], [9..10]`. Sizes: `(3, 3, 3, 2)`. `g_max = 3`. `num_top_groups = 4`.

Each top rep sends 4 blocks, each padded to `g_max² = 9` slots — `4 × 9 = 36` slots outgoing per rep. The size-2 group's rep (site 9) sends only `2 × 3 = 6` real slots per block to size-3 destinations and `2 × 2 = 4` real slots to itself; padding waste at site 9 across all 4 outgoing blocks is `(9-6) + (9-6) + (9-6) + (9-4) = 14` slots — bounded.

On receive, the rep at site 0 (`my_group.size = 3`) reconstructs `Col_0` as a `3 × 11` matrix:

| `src` | source group | `(src_group.size, my_group.size)` real region | fills `Col_0[0..2][·]` columns |
|---|---|---|---|
| 0 | `[0..2]` | `3 × 3` | `0..2` |
| 1 | `[3..5]` | `3 × 3` | `3..5` |
| 2 | `[6..8]` | `3 × 3` | `6..8` |
| 3 | `[9..10]` | `2 × 3` | `9..10` |

The reconstruction loop's bounds `(src_group.size, my_group.size)` are exactly the real region of each padded block; padding slots are never read.

The rep at site 9 (`my_group.size = 2`) reconstructs a `2 × 11` matrix using the symmetric pattern; from each source `src` it reads a `(src_group.size, 2)` real region.

## 8. Phase 3 — intra-subtree scatter

Phase 3 mirrors Phase 1 in structure.

**Top representatives.** Helper that mirrors `scatter_to`'s body but skips the cross-top first call:

```cpp
template <typename T>
hpx::future<std::vector<T>>
subtree_scatter_at_top_rep(
    hierarchical_communicator const& communicators,
    std::vector<std::vector<T>>&& subtree_outputs,  // Col_k, shape |G_k| × N
    generation_arg generation);
```

The implementation mirrors `scatter_to` hierarchical (§2.3), with two differences:

1. The initial cross-top scatter on `communicators.get(0)` is omitted; the chain starts at `communicators.get(1)` with `subtree_outputs` directly.
2. Inside the subtree, `this_site_arg(0)` is correct uniformly (a top rep is rank 0 in every subtree-internal communicator it appears on, §2.3).

The exact data shaping at each level — how `|G_k|` rows are partitioned across the `arity` children of each subtree-internal communicator — mirrors `detail::scatter_data`'s existing logic in `scatter.hpp`; the helper does not re-derive it.

Edge cases:

- `communicators.size() == 2`: the only subtree-internal communicator below the top is the terminal one. The helper performs a single scatter at `communicators.back()` directly. This includes the singleton-subtree case (top group of size 1, e.g., site 2 in `N=5, arity=4`) — there `subtree_outputs.size() == 1` and the terminal communicator has 1 participant, so the scatter is degenerate but correct.
- `communicators.size() == 1`: not reachable here. The §10 short-circuit at the entry point handles flat-fallback / `num_sites <= arity` before any helper is dispatched.

**Non-representatives.** The existing public hierarchical `scatter_from` works *as is* for Phase 3, for the same reason `gather_there` worked for Phase 1: on a non-top-rep, `communicators.get(0)` is subtree-internal, so the initial `scatter_from(communicators.get(0), …)` receives within the subtree from the next-level-up representative.

The Phase 3 internal generation is `generation_arg(3 * generation)` everywhere.

## 9. Generation handling — uniform stride three

Closes #7200 Q5.

Every hierarchical collective call on a `hierarchical_communicator` reserves three internal generation slots `[3k-2, 3k-1, 3k]` for user generation `k`:

| collective | slots used | last call's generation |
|---|---|---|
| `all_reduce`, `all_gather` (two-phase) | `3k-2, 3k` (skip `3k-1`) | `3k` |
| `all_to_all` (three-phase) | `3k-2, 3k-1, 3k` | `3k` |

Both end at `3k`, so consecutive user generations chain cleanly: user generation `k+1` starts at `3(k+1)-2 = 3k+1`, immediately after the prior call's last slot.

### 9.1 Why no dummy round-trip is needed

The internal gate's `next_generation` (§2.5) accepts any `new_generation >= generation_` and post-increments. The two-phase skip falls out of this:

- Two-phase phase 1 at `3k-2`: gate accepts (`3k-2 >= prev_last + 1 = 3(k-1) + 1 = 3k-2` for the second call onward; trivially true for the first call). Gate advances to `3k-1`.
- Two-phase phase 2 at `3k`: gate accepts (`3k >= 3k-1`). Gate advances to `3k+1`.
- The slot `3k-1` is **consumed implicitly** by the `>=` accept — no dummy collective, no extra wire round-trip. Per `@hkaiser` in #7200: *"instead of adding another round trip simply send the required increment with the last operation on a communicator."*

For three-phase `all_to_all` the same mechanism is invoked with no skip: phases use `3k-2, 3k-1, 3k` consecutively, the gate post-increments at each step, and the final state is `3k+1`. Both shapes leave the gate at the same state, so they can be freely interleaved on the same communicator across user generations.

### 9.2 What the user sees

The user generation contract is unchanged: each call on a shared `hierarchical_communicator` must use a strictly greater `generation` than the previous call. Stride-three is an *internal* invariant; it is invisible at the API boundary.

What the user *gains* is predictability across operation kinds. A sequence

```text
all_gather (gen=1)  ->  internal slots 1, _, 3   (last = 3)
all_to_all (gen=2)  ->  internal slots 4, 5, 6   (last = 6)
all_reduce (gen=3)  ->  internal slots 7, _, 9   (last = 9)
```

never collides on the underlying communicators' generation namespaces, regardless of how two-phase and three-phase calls are interleaved.

### 9.3 What this is *not*

Stride-three does not relax the user-visible monotonicity rule. Two distinct calls on the same `hierarchical_communicator` with the same user generation are still an error; the prohibition is independent of stride.

### 9.4 Prerequisite PR scope (§14, step 0)

A separate PR before all_to_all:

- `all_reduce.hpp:459-460`: change `2 * generation - 1, 2 * generation` to `3 * generation - 2, 3 * generation`.
- `all_gather.hpp:420-421`: same.
- `all_reduce.hpp:451` and `all_gather.hpp:412`: change the `bad_parameter` strings from "the 2k/2k+1 internal mapping" to "the 3k-2/3k internal mapping" (the existing strings already describe the wrong arithmetic — pre-existing slip, fixed in passing).
- A regression test mixing `all_reduce` and `all_gather` calls at consecutive user generations on the same `hierarchical_communicator`, asserting all calls succeed. The cross-collective regression that includes `all_to_all` is added when the all_to_all PR lands.

No changes to `next_generation`, `handle_data`, or any other internal communicator API.

## 10. Flat fallback

At entry to hierarchical `all_to_all`:

```cpp
if (communicators.size() == 1)
{
    auto [c, site] = communicators[0];
    return all_to_all(c, HPX_MOVE(local_result), site, generation);
}
```

This handles two cases uniformly:

1. The factory took the explicit flat-fallback path from PR #7193 (`num_sites < threshold`).
2. The tree construction reached the terminal flat case immediately because `num_sites <= arity`.

Flat `all_to_all`'s validation and exception behavior (size mismatch, `generation == 0`) are inherited unchanged.

This mirrors the existing `scatter_from` and `scatter_to` hierarchical `size() == 1` short-circuits at `scatter.hpp:547` and `scatter.hpp:713`.

## 11. Memory behavior

Total logical data volume is unavoidable: every site contributes `N` values and receives `N` values, irrespective of the algorithm.

The pressure point is a top representative. In a straightforward v1 implementation a rep transiently holds up to four buffers of order `g × N` elements (with `g ≈ N / arity` for balanced trees): `Row_k`, outgoing padded blocks, received padded blocks, and `Col_k`. Aggressive use of `HPX_MOVE` keeps two of these alive at a time in steady state; peak usage stays in `O(g · N · sizeof(T))`.

Per-rep peak (rough upper bound, balanced case):

```text
peak_bytes ≈ (2/arity + 1/arity²) · N² · m
```

where `m = sizeof(payload-element)`. For `N = 256, arity = 4, m = 1 MiB` this is approximately `36 GiB` at a single rep, which is unusable on most nodes.

**v1 disposition (closes #7200 Q6):** no second fallback threshold in v1. Document the budget; users size `arity` to keep per-rep peak within node memory. `@hkaiser`: *"Let's cross that bridge when we're there."*

Future work that addresses this — listed here for visibility, not for v1 — includes ragged blocks instead of padded (eliminates the `g_max² - g_src · g_dst` waste per block), streaming Phase 2 → Phase 3 to avoid materializing the full `Col_k`, a payload-byte-keyed second fallback threshold, an `all_to_allv` based on `channel_communicator`, and multi-leader variants for large groups. None are v1.

## 12. Tests

Tests live alongside the existing hierarchical collective tests and follow the same launch-`num_sites`-tasks-from-locality-0 pattern (compare the existing flat `all_to_all` test at `concurrent_collectives.cpp:189-200`).

### 12.1 Transpose identity

Site `i` contributes `In[i][j] = encode(i, j)` for an injective deterministic encode such as

```cpp
auto encode = [](std::size_t i, std::size_t j, std::size_t C) {
    return i * C + j;
};
```

with `C > max_num_sites` to keep the encoding injective. After the collective, site `i` checks `Out[i][j] == encode(j, i)` for all `j`. This catches every transpose error.

### 12.2 Local multi-site coverage

```text
arity 2: N = 2, 3, 4, 5, 6, 7, 8, 9, 11, 15
arity 4: N = 5, 6, 7, 9, 10, 11, 13, 15
```

`flat_fallback_threshold_arg(0)` forces the tree path except in the terminal-flat case where `num_sites <= arity` (§2.2); a high threshold (e.g. `1024`) forces the explicit flat-fallback path. The `N = 11, arity = 4` case is the worked example in §7.4 and exercises the most non-trivial padding/reconstruction path.

### 12.3 Distributed coverage

The existing CMake `LOCALITIES 2` pattern is sufficient for API surface, fallback path, and basic distributed correctness, but it does *not* exercise a real top-level hierarchy: at `N=2, arity=2` the recursion's terminal condition (§2.2) fires immediately (`right - left = 1 < arity = 2`), producing a `hierarchical_communicator` of `size() == 1` and dispatching through the §10 fallback regardless of the `flat_fallback_threshold` value.

Real hierarchical multi-level behavior is therefore covered by the local multi-site tests in §12.2 (which run `num_sites` HPX tasks from a single locality) and, optionally, by a higher-locality distributed target if CI cost permits — `LOCALITIES 4` with `arity 2` is enough to produce a two-level tree.

### 12.4 Bad-input path

Mirror flat `all_to_all`: if any site's contribution is not of length `num_sites`, the operation completes exceptionally with `bad_parameter`. Same for `generation == 0` and for the default-generation case (§4).

### 12.5 Generation regression — cross-collective stride-three

On one `hierarchical_communicator`:

```text
all_gather  (gen=1)
all_to_all  (gen=2)
all_reduce  (gen=3)
all_to_all  (gen=4)
```

All calls must succeed.

The failure mode this guards against is mixing operation strides on a shared communicator. Under the *old* two-phase mapping (`2k-1, 2k`), `all_gather(gen=1)` would consume internal slots `1, 2`, leaving the gate expecting at least `3` for the next call. A stride-three `all_to_all(gen=2)` would consume slots `4, 5, 6`, leaving the gate expecting at least `7`. A subsequent old-mapped `all_reduce(gen=3)` would attempt to start at slot `2*3 - 1 = 5`, which is now *behind* the gate (`5 < 7`) — `next_generation`'s `>=` check rejects, the call fails. The stride-three prerequisite eliminates this by making `all_reduce(gen=3)` start at slot `3*3 - 2 = 7`, which the gate accepts.

This regression is the test that *justifies* the §9 design. The two-collective version (without `all_to_all`) lives in the prerequisite PR (§14, step 0); the four-call version above is added when all_to_all lands.

### 12.6 Helper unit tests

`detail::get_top_level_groups` and `detail::classify_site` get a small unit test against a hardcoded partition table, covering at minimum:

- balanced cases: `(N=8, arity=2)`, `(N=16, arity=4)`;
- non-balanced cases: `(N=11, arity=4)` (the §7.4 example, sizes `3,3,3,2`), `(N=5, arity=4)` (sizes `2,1,1,1` — three top reps with singleton top-level groups).

These tests live in the prerequisite PR (§14, step 1).

## 13. Benchmarks

Benchmarking comes after correctness. The first extension follows the existing `benchmark_collectives_test` style:

- flat vs hierarchical `all_to_all`;
- arity 2, 4 (and 8 if cluster geometry permits);
- fallback threshold `0` for forced tree mode and the default for normal behavior;
- payload sizes from scalar to large vectors;
- threads-per-locality sweep, matching the previous fallback-benchmark discussion.

Per-phase timing is useful but should not block the correctness PR; if added, it stays in the benchmark harness, not the collective.

## 14. Implementation sequence

Each step is a standalone PR.

**Step 0 — prerequisite: stride-three arithmetic.** Update `all_reduce.hpp` and `all_gather.hpp` from `(2k-1, 2k)` to `(3k-2, 3k)`. Fix the `bad_parameter` strings. Add the two-collective generation regression test. No internal communicator API change.

**Step 1 — prerequisite: shared top-level partition helper.** Introduce `detail::get_top_level_groups` and `detail::classify_site`. Use `get_top_level_groups` in the top frame of `recursively_fill_communicators` *without changing the produced communicator names, group boundaries, or site ranks* — this PR is a refactor, not a behavior change. Add §12.6's unit tests.

**Step 2 — `all_to_all` fallback and validation.** Hierarchical entry point with the §10 `size() == 1` shortcut, malformed-size validation, and `generation == 0` rejection. Add §12.4's tests.

**Step 3 — `subtree_gather_at_top_rep` helper (Phase 1).** Plus a unit test that exercises only Phase 1 against a known input.

**Step 4 — Phase 2 representative exchange.** Padded block builder, flat `all_to_all<vector<T>>` over the top communicator, and `Col_k` reconstruction using §5's helpers.

**Step 5 — `subtree_scatter_at_top_rep` helper (Phase 3).** Plus a unit test that exercises only Phase 3 against a known input.

**Step 6 — End-to-end transpose tests.** §12.1 + §12.2 + §12.3 + the four-call version of §12.5.

**Step 7 — Benchmark wiring.** Per §13.

Steps 0 and 1 are PR-able immediately and depend on nothing in this design. Steps 2–6 land as one feature PR or as a small chain.

## 15. Out of scope for v1

- Separate communicator families per collective operation (rejected in #7200; would prevent shared communicator identity).
- Second fallback threshold keyed on payload bytes (deferred per #7200; ship v1 with a documented memory budget instead).
- Custom `all_to_allv` over `channel_communicator`.
- Ragged (non-padded) blocks. Two benefits, both deferred to follow-up: (a) eliminates the `g_max² - g_src · g_dst` waste per block, and (b) lifts the v1 default-constructibility constraint on `T` (§7.1), since blocks would carry only real elements. This is the natural near-term Phase-2.5 follow-up.
- Streaming Phase 2 → Phase 3 to avoid materializing the full `Col_k`.
- GPU-aware payload handling.
- Phase overlap / pipelining.
- Multi-leader representative schemes.
- Performance-driven changes to the default `flat_fallback_threshold` value.

These are valid follow-ups; including any one in v1 would force a too-large initial PR.

## 16. Pitfalls

1. **`communicators.get(0)` is the top-of-tree communicator only on top reps.** On any non-top-rep it is the highest *subtree-internal* communicator. Implementations that branch on it must compute `is_top_rep` from `(num_sites, arity, this_site)` via §5's helper, not from `communicators[0]` alone.
2. **Public hierarchical `gather_here` on a top rep ends with a cross-top gather**; Phase 1 on top reps must use `subtree_gather_at_top_rep`. Public `gather_there` on non-top-reps is fine as-is.
3. **Public hierarchical `scatter_to` on a top rep starts with a cross-top scatter**; Phase 3 on top reps must use `subtree_scatter_at_top_rep`. Public `scatter_from` on non-top-reps is fine as-is.
4. **Padding is not a serialization requirement.** `all_to_all<vector<T>>` carries variable-length inner vectors fine. Padding is purely v1's receiver-side reconstruction simplicity.
5. **Stride-three needs no internal communicator API change.** `next_generation`'s `>=` accept + post-increment carries the implicit skip; do not introduce a dummy collective for the unused middle slot.
6. **Reusing the same user generation for two distinct calls on the same `hierarchical_communicator` is still an error** after the stride-three change. Stride-three makes the *internal* generation footprint per call uniform; it does not relax the user contract.
7. **Receiver-side `Col_k` reconstruction in Phase 2 needs the *source* group's actual size** to crop padding correctly. Get it from `detail::get_top_level_groups(num_sites, arity)[src]`; do not assume balanced groups.
8. **Top-rep detection comes from the partition helper, not from the communicator path.** Use `detail::classify_site(get_top_level_groups(...), this_site).is_representative` to decide whether to dispatch to the subtree gather/scatter helpers. Inferring top-rep-ness from `communicators.size()` or from positional checks on `communicators[0]` is fragile: a top rep with a singleton top-level group has `communicators.size() == 2` (top comm + 1-site terminal), not 1; the only `size() == 1` case is the §10 short-circuit, which fires before any top-rep dispatch.
9. **Do not duplicate the partitioning rule in two places.** Step 1 of §14 introduces `detail::get_top_level_groups` precisely to keep the factory and the helper in lockstep.

## 17. Summary

```text
if hierarchical communicator has one underlying communicator:
    delegate to flat all_to_all
    (covers the explicit fallback path AND the case num_sites <= arity)

else:
    Phase 1 (gen 3k-2): each top-level subtree gathers to its representative
        top reps:        subtree_gather_at_top_rep helper (loop, no cross-top)
        non-top-reps:    public hierarchical gather_there (works as-is)

    Phase 2 (gen 3k-1): top reps run flat all_to_all<vector<T>> on the top
                        communicator with padded |G_k| × |G_dst| blocks;
                        receivers crop padding using get_top_level_groups

    Phase 3 (gen 3k):   each top rep scatters Col_k down its subtree
        top reps:        subtree_scatter_at_top_rep helper (start at comm[1])
        non-top-reps:    public hierarchical scatter_from (works as-is)
```

Correctness first. The two prerequisite PRs (stride-three arithmetic, shared partition helper) are small, mechanical, and reviewable in isolation; they unlock the `all_to_all` PR sequence without coupling.

## References

- Strack, Zeil, Kaiser, Pflüger. *Hierarchical Collective Operations for the Asynchronous Many-Task Runtime HPX.* 16th International Parallel Tools Workshop (IPTW), 2025.
- Träff, Rougier. *MPI Collectives and Datatypes for Hierarchical All-to-all Communication.* EuroMPI 2014.
- Chochia, Solt, Hursey. *Applying On-Node Aggregation Methods to MPI Alltoall Collectives: Matrix Block Aggregation Algorithm.* EuroMPI/USA 2022.
- Bienz, Olson, Gropp. *Node-Aware Improvements to Allreduce.* ExaMPI Workshop, SC 2019. arXiv:1910.09650.
- HPX PR #7160 (hierarchical `all_reduce` / `all_gather`, merged).
- HPX PR #7193 (flat-fallback factory threshold, in review).
- HPX Discussion #7200 (architectural Q&A for this design).
