//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/affinity/parse_affinity_options.hpp>
#include <hpx/assert.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/topology/topology.hpp>

#include <hpx/affinity/detail/partlit.hpp>

#include <hwloc.h>

#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/include/std_pair.hpp>
#include <boost/spirit/home/x3/auxiliary.hpp>
#include <boost/spirit/home/x3/char.hpp>
#include <boost/spirit/home/x3/core.hpp>
#include <boost/spirit/home/x3/nonterminal.hpp>
#include <boost/spirit/home/x3/numeric.hpp>
#include <boost/spirit/home/x3/operator.hpp>
#include <boost/spirit/home/x3/string.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
// clang-format off
BOOST_FUSION_ADAPT_STRUCT(
    hpx::threads::detail::spec_type,
    type_,
    index_bounds_
)
// clang-format on

namespace {

    ///////////////////////////////////////////////////////////////////////////
    //
    //    mappings:
    //        distribution
    //        mapping(;mapping)*
    //
    //    distribution:
    //        compact
    //        scatter
    //        balanced
    //        numa-balanced
    //
    //    mapping:
    //        thread-spec=pu-specs
    //
    //    thread-spec:
    //        t:int
    //        t:int-int
    //        t:all
    //
    //    pu-specs:
    //        pu-spec(.pu-spec)*
    //
    //    pu-spec:
    //        type:range-specs
    //        ~pu-spec
    //
    //    range-specs:
    //        range-spec(,range-spec)*
    //
    //    range-spec:
    //        int
    //        int-int
    //        all
    //
    //    type:
    //        socket | numanode
    //        core
    //        pu
    //
    namespace x3 = boost::spirit::x3;
    using namespace hpx::threads::detail;

    x3::rule<class mappings, mappings_type> mappings = "mappings";
    x3::rule<class distribution, distribution_type> distribution =
        "distribution";
    x3::rule<class mapping, full_mapping_type> mapping = "mapping";
    x3::rule<class thread_spec, spec_type> thread_spec = "thread_spec";
    x3::rule<class pu_specs, mapping_type> pu_specs = "pu_specs";
    x3::rule<class socket_spec, spec_type> socket_spec = "socket_spec";
    x3::rule<class core_spec, spec_type> core_spec = "core_spec";
    x3::rule<class pu_spec, spec_type> pu_spec = "pu_spec";
    x3::rule<class specs, bounds_type> specs = "specs";
    x3::rule<class spec, bounds_type> spec = "spec";

    auto mappings_def = distribution | (mapping % ';');

    auto mapping_def = thread_spec >> '=' >> pu_specs;

    // clang-format off
    auto distribution_def =
            partlit("compact", compact)
        |   partlit("scatter", scatter)
        |   partlit("balanced", balanced)
        |   partlit("numa-balanced", numa_balanced)
        ;

    auto thread_spec_def =
            partlit("thread", spec_type::thread) >> ':' >>  specs
        ;

    auto pu_specs_def =
            (socket_spec >> core_spec >> pu_spec)
//        |   ('~' >> pu_spec)
        ;

    auto socket_spec_def =
            (partlit("socket", spec_type::socket) >> ':' >> specs)
        |   (partlit("numanode", spec_type::numanode) >> ':' >> specs)
        |   x3::attr(spec_type(spec_type::unknown))
        ;

    auto core_spec_def =
           (-x3::lit('.') >> partlit("core", spec_type::core) >> ':' >> specs)
        |   x3::attr(spec_type(spec_type::unknown))
        ;

    auto pu_spec_def =
           (-x3::lit('.') >> partlit("pu", spec_type::pu) >> ':' >> specs)
        |   x3::attr(spec_type(spec_type::unknown))
        ;

    auto specs_def = spec % ',';

    auto spec_def =
            (x3::uint_ >> -(x3::int_))
        |   partlit("all", bounds_type{spec_type::all_entities()})
        ;
    // clang-format on

    BOOST_SPIRIT_DEFINE(mappings, distribution, mapping, thread_spec, pu_specs,
        socket_spec, core_spec, pu_spec, specs, spec)

}    // namespace

namespace hpx { namespace threads { namespace detail {
    static char const* const type_names[] = {
        "unknown", "thread", "socket", "numanode", "core", "pu"};

    char const* spec_type::type_name(spec_type::type t)
    {
        if (t < spec_type::unknown || t > spec_type::pu)
            return type_names[0];
        return type_names[t];
    }

    template <typename Iterator>
    inline bool parse(Iterator& begin, Iterator end, mappings_type& m)
    {
        return x3::parse(begin, end, mappings, m);
    }

    ///////////////////////////////////////////////////////////////////////////
    void parse_mappings(
        std::string const& spec, mappings_type& mappings, error_code& ec)
    {
        std::string::const_iterator begin = spec.begin();
        if (!detail::parse(begin, spec.end(), mappings) || begin != spec.end())
        {
            HPX_THROWS_IF(ec, bad_parameter, "parse_affinity_options",
                "failed to parse affinity specification: " + spec);
            return;
        }

        if (&ec != &throws)
            ec = make_success_code();
    }

    ///////////////////////////////////////////////////////////////////////////
    bounds_type extract_bounds(
        spec_type const& m, std::size_t default_last, error_code& ec)
    {
        bounds_type result;

        if (m.index_bounds_.empty())
            return result;

        bounds_type::const_iterator first = m.index_bounds_.begin();
        bounds_type::const_iterator last = m.index_bounds_.end();

        while (first != last)
        {
            if (*first == spec_type::all_entities())
            {
                // bind all entities
                result.clear();
                for (std::size_t i = 0; i != default_last; ++i)
                    result.push_back(i);
                break;    // we will not get more than 'all'
            }

            bounds_type::const_iterator second = first;
            if (++second != last)
            {
                if (*second == 0 || *second == spec_type::all_entities())
                {
                    // one element only
                    if (default_last <= std::size_t(*first))
                    {
                        result.clear();
                        HPX_THROWS_IF(ec, bad_parameter, "extract_bounds",
                            "the resource id given is larger than the "
                            "number of existing resources");
                        return result;
                    }
                    result.push_back(*first);
                }
                else if (*second < 0)
                {
                    // all elements between min and -max
                    if (default_last <= std::size_t(-*second))
                    {
                        result.clear();
                        HPX_THROWS_IF(ec, bad_parameter, "extract_bounds",
                            "the upper limit given is larger than the "
                            "number of existing resources");
                        return result;
                    }
                    for (std::int64_t i = *first; i <= -*second; ++i)
                        result.push_back(i);
                }
                else
                {
                    // just min and max
                    if (default_last <= std::size_t(*second))
                    {
                        result.clear();
                        HPX_THROWS_IF(ec, bad_parameter, "extract_bounds",
                            "the upper limit given is larger than the "
                            "number of existing resources");
                        return result;
                    }
                    result.push_back(*first);
                    result.push_back(*second);
                }
                first = second;
            }
            else
            {
                // one element only
                if (default_last <= std::size_t(*first))
                {
                    result.clear();
                    HPX_THROWS_IF(ec, bad_parameter, "extract_bounds",
                        "the resource id given is larger than the number "
                        "of existing resources");
                    return result;
                }
                result.push_back(*first);
            }
            ++first;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////
    //                  index,       mask
    typedef hpx::tuple<std::size_t, mask_type> mask_info;

    inline std::size_t get_index(mask_info const& smi)
    {
        return hpx::get<0>(smi);
    }
    inline mask_cref_type get_mask(mask_info const& smi)
    {
        return hpx::get<1>(smi);
    }

    ///////////////////////////////////////////////////////////////////////////
    std::vector<mask_info> extract_socket_masks(
        topology const& t, bounds_type const& b)
    {
        std::vector<mask_info> masks;
        for (std::int64_t index : b)
        {
            masks.push_back(hpx::make_tuple(static_cast<std::size_t>(index),
                t.init_socket_affinity_mask_from_socket(
                    static_cast<std::size_t>(index))));
        }
        return masks;
    }

    std::vector<mask_info> extract_numanode_masks(
        topology const& t, bounds_type const& b)
    {
        std::vector<mask_info> masks;
        for (std::int64_t index : b)
        {
            masks.push_back(hpx::make_tuple(static_cast<std::size_t>(index),
                t.init_numa_node_affinity_mask_from_numa_node(
                    static_cast<std::size_t>(index))));
        }
        return masks;
    }

    mask_cref_type extract_machine_mask(topology const& t, error_code& ec)
    {
        return t.get_machine_affinity_mask(ec);
    }

    std::vector<mask_info> extract_socket_or_numanode_masks(
        topology const& t, spec_type const& s, error_code& ec)
    {
        switch (s.type_)
        {
        case spec_type::socket:
            // requested top level is a socket
            {
                std::size_t num_sockets = t.get_number_of_sockets();
                return extract_socket_masks(
                    t, extract_bounds(s, num_sockets, ec));
            }

        case spec_type::numanode:
            // requested top level is a NUMA node
            {
                std::size_t num_numanodes = t.get_number_of_numa_nodes();
                return extract_numanode_masks(
                    t, extract_bounds(s, num_numanodes, ec));
            }

        case spec_type::unknown:
        {
            std::vector<mask_info> masks;
            masks.push_back(
                hpx::make_tuple(std::size_t(-1), extract_machine_mask(t, ec)));
            return masks;
        }

        default:
            HPX_THROWS_IF(ec, bad_parameter, "extract_socket_or_numanode_mask",
                "unexpected specification type {}",
                spec_type::type_name(s.type_));
            break;
        }

        return std::vector<mask_info>();
    }

    std::vector<mask_info> extract_core_masks(topology const& t,
        spec_type const& s, std::size_t socket, mask_cref_type socket_mask,
        error_code& ec)
    {
        std::vector<mask_info> masks;

        switch (s.type_)
        {
        case spec_type::core:
        {
            std::size_t base = 0;
            std::size_t num_cores = 0;

            if (socket != std::size_t(-1))
            {
                for (std::size_t i = 0; i != socket; ++i)
                {
                    // The number of NUMA nodes might be zero if there hwloc
                    // doesn't detect a NUMA domain. This might be the case
                    // when there is no NUMA support configured, or when
                    // there are just sockets, but no direct numa domains.
                    // The bind description might relate to sockets, not
                    // NUMA domains.
                    if (t.get_number_of_numa_nodes() == 0)
                        base += t.get_number_of_socket_cores(i);
                    else
                        base += t.get_number_of_numa_node_cores(i);
                }
                if (t.get_number_of_numa_nodes() == 0)
                    num_cores = t.get_number_of_socket_cores(socket);
                else
                    num_cores = t.get_number_of_numa_node_cores(socket);
            }
            else
            {
                num_cores = t.get_number_of_cores();
            }

            bounds_type bounds = extract_bounds(s, num_cores, ec);
            if (ec)
                break;

            for (std::int64_t index : bounds)
            {
                mask_type mask = t.init_core_affinity_mask_from_core(
                    static_cast<std::size_t>(index + base));
                masks.push_back(hpx::make_tuple(
                    static_cast<std::size_t>(index), mask & socket_mask));
            }
        }
        break;

        case spec_type::unknown:
        {
            mask_type mask = extract_machine_mask(t, ec);
            masks.push_back(
                hpx::make_tuple(std::size_t(-1), mask & socket_mask));
        }
        break;

        default:
            HPX_THROWS_IF(ec, bad_parameter, "extract_core_mask",
                "unexpected specification type {}",
                spec_type::type_name(s.type_));
            break;
        }

        return masks;
    }

    std::vector<mask_info> extract_pu_masks(topology const& t,
        spec_type const& s, std::size_t socket, std::size_t core,
        mask_cref_type core_mask, error_code& ec)
    {
        std::vector<mask_info> masks;

        switch (s.type_)
        {
        case spec_type::pu:
        {
            std::size_t num_pus = 0;
            std::size_t socket_base = 0;
            if (std::size_t(-1) != socket)
            {
                // core number is relative to socket
                for (std::size_t i = 0; i != socket; ++i)
                {
                    if (t.get_number_of_numa_nodes() == 0)
                        socket_base += t.get_number_of_socket_cores(i);
                    else
                        socket_base += t.get_number_of_numa_node_cores(i);
                }
            }

            if (std::size_t(-1) != core)
            {
                num_pus = t.get_number_of_core_pus(core);
            }
            else
            {
                num_pus = t.get_number_of_pus();
            }
            bounds_type bounds = extract_bounds(s, num_pus, ec);
            if (ec)
                break;

            std::size_t num_cores = t.get_number_of_cores();
            for (std::int64_t index : bounds)
            {
                std::size_t base_core = socket_base;
                if (std::size_t(-1) != core)
                {
                    base_core += core;
                }
                else
                {
                    // find core the given pu belongs to
                    std::size_t base = 0;
                    for (/**/; base_core < num_cores; ++base_core)
                    {
                        std::size_t num_core_pus =
                            t.get_number_of_core_pus(base_core);
                        if (base + num_core_pus > std::size_t(index))
                            break;
                        base += num_core_pus;
                    }
                }

                mask_type mask = t.init_thread_affinity_mask(
                    base_core, static_cast<std::size_t>(index));
                masks.push_back(hpx::make_tuple(
                    static_cast<std::size_t>(index), mask & core_mask));
            }
        }
        break;

        case spec_type::unknown:
        {
            mask_type mask = extract_machine_mask(t, ec);
            masks.push_back(hpx::make_tuple(std::size_t(-1), mask & core_mask));
        }
        break;

        default:
            HPX_THROWS_IF(ec, bad_parameter, "extract_pu_mask",
                "unexpected specification type {}",
                spec_type::type_name(s.type_));
            break;
        }

        return masks;
    }

    ///////////////////////////////////////////////////////////////////////////
    // sanity checks
    void mappings_sanity_checks(full_mapping_type& fmt, std::size_t /* size */,
        bounds_type const& b, error_code& ec)
    {
        mapping_type& m = fmt.second;
        if (m.size() != 3)
        {
            HPX_THROWS_IF(ec, bad_parameter, "decode_mapping",
                "bad size of mappings specification array");
            return;
        }

        if (b.begin() == b.end())
        {
            HPX_THROWS_IF(ec, bad_parameter, "decode_mapping",
                "no {1} mapping bounds are specified",
                spec_type::type_name(fmt.first.type_));
            return;
        }

        if (&ec != &throws)
            ec = make_success_code();
    }

    // for each given core-mask extract all required pu-masks
    void extract_pu_affinities(topology const& t,
        std::vector<spec_type> const& specs, std::size_t socket,
        std::vector<mask_info> const& core_masks,
        std::vector<mask_type>& affinities, error_code& ec)
    {
        // get the core masks for each of the sockets
        for (mask_info const& cmi : core_masks)
        {
            if (get_index(cmi) == std::size_t(-1))
            {
                // all cores
                if (specs[2].type_ == spec_type::unknown)
                {
                    // no pu information
                    affinities.push_back(get_mask(cmi));
                }
                else
                {
                    // handle pu information in the absence of core information
                    std::vector<mask_info> pu_masks = extract_pu_masks(t,
                        specs[2], socket, std::size_t(-1), get_mask(cmi), ec);
                    if (ec)
                        break;

                    for (mask_info const& pmi : pu_masks)
                    {
                        affinities.push_back(get_mask(pmi));
                    }
                }
                break;
            }
            else
            {
                // just this core
                std::vector<mask_info> pu_masks = extract_pu_masks(
                    t, specs[2], socket, get_index(cmi), get_mask(cmi), ec);
                if (ec)
                    break;

                for (mask_info const& pmi : pu_masks)
                {
                    affinities.push_back(get_mask(pmi));
                }
            }
        }
    }

    // for each given socket-mask extract all required pu-masks
    void extract_core_affinities(topology const& t,
        std::vector<spec_type> const& specs,
        std::vector<mask_info> const& socket_masks,
        std::vector<mask_type>& affinities, error_code& ec)
    {
        // get the core masks for each of the sockets
        for (mask_info const& smi : socket_masks)
        {
            if (get_index(smi) == std::size_t(-1))
            {
                // all NUMA domains
                if (specs[1].type_ == spec_type::unknown)
                {
                    // no core information
                    if (specs[2].type_ == spec_type::unknown)
                    {
                        // no pu information
                        affinities.push_back(get_mask(smi));
                    }
                    else
                    {
                        // handle pu information in the absence of core/socket
                        std::vector<mask_info> pu_masks =
                            extract_pu_masks(t, specs[2], std::size_t(-1),
                                std::size_t(-1), get_mask(smi), ec);
                        if (ec)
                            break;

                        for (mask_info const& pmi : pu_masks)
                        {
                            affinities.push_back(get_mask(pmi));
                        }
                    }
                }
                else
                {
                    // no socket given, assume cores are numbered for whole
                    // machine
                    if (specs[2].type_ == spec_type::unknown)
                    {
                        // no pu information
                        std::vector<mask_info> core_masks = extract_core_masks(
                            t, specs[1], std::size_t(-1), get_mask(smi), ec);
                        if (ec)
                            break;

                        for (mask_info const& cmi : core_masks)
                        {
                            affinities.push_back(get_mask(cmi));
                        }
                    }
                    else
                    {
                        std::vector<mask_info> core_masks = extract_core_masks(
                            t, specs[1], std::size_t(-1), get_mask(smi), ec);
                        if (ec)
                            break;

                        // get the pu masks (i.e. overall affinity masks) for
                        // all of the core masks
                        extract_pu_affinities(t, specs, std::size_t(-1),
                            core_masks, affinities, ec);
                        if (ec)
                            break;
                    }
                }
                break;
            }
            else
            {
                std::vector<mask_info> core_masks = extract_core_masks(
                    t, specs[1], get_index(smi), get_mask(smi), ec);
                if (ec)
                    break;

                // get the pu masks (i.e. overall affinity masks) for
                // all of the core masks
                extract_pu_affinities(
                    t, specs, get_index(smi), core_masks, affinities, ec);
                if (ec)
                    break;
            }
        }
    }

    void decode_mappings(topology const& t, full_mapping_type& m,
        std::vector<mask_type>& affinities, std::size_t num_threads,
        error_code& ec)
    {
        // The core numbers are interpreted differently depending on whether a
        // socket/numanode is given or not. If no socket(s) is(are) given, then
        // the core numbering covers the whole locality, otherwise the core
        // numbering is relative to the given socket.

        // generate overall masks for each of the given sockets
        std::vector<mask_info> socket_masks =
            extract_socket_or_numanode_masks(t, m.second[0], ec);

        HPX_ASSERT(!socket_masks.empty());

        extract_core_affinities(t, m.second, socket_masks, affinities, ec);

        // special case, all threads share the same options
        if (affinities.size() == 1 && num_threads > 1)
        {
            affinities.resize(num_threads, affinities[0]);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    bool pu_in_process_mask(bool use_process_mask, topology& t,
        std::size_t num_core, std::size_t num_pu)
    {
        if (!use_process_mask)
        {
            return true;
        }

        threads::mask_type proc_mask = t.get_cpubind_mask();
        threads::mask_type pu_mask =
            t.init_thread_affinity_mask(num_core, num_pu);

        return threads::bit_and(proc_mask, pu_mask);
    }

    void check_num_threads(bool use_process_mask, topology& t,
        std::size_t num_threads, error_code& ec)
    {
        if (use_process_mask)
        {
            threads::mask_type proc_mask = t.get_cpubind_mask();
            std::size_t num_pus_proc_mask = threads::count(proc_mask);

            if (num_threads > num_pus_proc_mask)
            {
                HPX_THROWS_IF(ec, bad_parameter, "check_num_threads",
                    "specified number of threads ({1}) is larger than number "
                    "of processing units available in process mask ({2})",
                    num_threads, num_pus_proc_mask);
            }
        }
        else
        {
            std::size_t num_threads_available = threads::hardware_concurrency();

            if (num_threads > num_threads_available)
            {
                HPX_THROWS_IF(ec, bad_parameter, "check_num_threads",
                    "specified number of threads ({1}) is larger than number "
                    "of available processing units ({2})",
                    num_threads, num_threads_available);
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    void decode_compact_distribution(topology& t,
        std::vector<mask_type>& affinities, std::size_t used_cores,
        std::size_t max_cores, std::vector<std::size_t>& num_pus,
        bool use_process_mask, error_code& ec)
    {
        std::size_t num_threads = affinities.size();

        check_num_threads(use_process_mask, t, num_threads, ec);

        if (use_process_mask)
        {
            used_cores = 0;
            max_cores = t.get_number_of_cores();
        }

        std::size_t num_cores = (std::min)(max_cores, t.get_number_of_cores());
        num_pus.resize(num_threads);

        for (std::size_t num_thread = 0; num_thread < num_threads; /**/)
        {
            for (std::size_t num_core = 0; num_core < num_cores; ++num_core)
            {
                std::size_t num_core_pus =
                    t.get_number_of_core_pus(num_core + used_cores);
                for (std::size_t num_pu = 0; num_pu < num_core_pus; ++num_pu)
                {
                    if (!pu_in_process_mask(
                            use_process_mask, t, num_core, num_pu))
                    {
                        continue;
                    }

                    if (any(affinities[num_thread]))
                    {
                        HPX_THROWS_IF(ec, bad_parameter,
                            "decode_compact_distribution",
                            "affinity mask for thread {1} has already been set",
                            num_thread);
                        return;
                    }

                    num_pus[num_thread] =
                        t.get_pu_number(num_core + used_cores, num_pu);
                    affinities[num_thread] = t.init_thread_affinity_mask(
                        num_core + used_cores, num_pu);

                    if (++num_thread == num_threads)
                        return;
                }
            }
        }
    }

    void decode_scatter_distribution(topology& t,
        std::vector<mask_type>& affinities, std::size_t used_cores,
        std::size_t max_cores, std::vector<std::size_t>& num_pus,
        bool use_process_mask, error_code& ec)
    {
        std::size_t num_threads = affinities.size();

        check_num_threads(use_process_mask, t, num_threads, ec);

        if (use_process_mask)
        {
            used_cores = 0;
            max_cores = t.get_number_of_cores();
        }

        std::size_t num_cores = (std::min)(max_cores, t.get_number_of_cores());

        std::vector<std::size_t> next_pu_index(num_cores, 0);
        num_pus.resize(num_threads);

        for (std::size_t num_thread = 0; num_thread < num_threads; /**/)
        {
            for (std::size_t num_core = 0; num_core < num_cores; ++num_core)
            {
                if (any(affinities[num_thread]))
                {
                    HPX_THROWS_IF(ec, bad_parameter,
                        "decode_scatter_distribution",
                        "affinity mask for thread {1} has already been set",
                        num_thread);
                    return;
                }

                std::size_t num_core_pus = t.get_number_of_core_pus(num_core);
                std::size_t pu_index = next_pu_index[num_core];
                bool use_pu = false;

                // Find the next PU on this core which is in the process mask
                while (pu_index < num_core_pus)
                {
                    use_pu = pu_in_process_mask(
                        use_process_mask, t, num_core, pu_index);
                    ++pu_index;

                    if (use_pu)
                    {
                        break;
                    }
                }

                next_pu_index[num_core] = pu_index;

                if (!use_pu)
                {
                    continue;
                }

                num_pus[num_thread] = t.get_pu_number(
                    num_core + used_cores, next_pu_index[num_core] - 1);
                affinities[num_thread] = t.init_thread_affinity_mask(
                    num_core + used_cores, next_pu_index[num_core] - 1);

                if (++num_thread == num_threads)
                    return;
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    void decode_balanced_distribution(topology& t,
        std::vector<mask_type>& affinities, std::size_t used_cores,
        std::size_t max_cores, std::vector<std::size_t>& num_pus,
        bool use_process_mask, error_code& ec)
    {
        std::size_t num_threads = affinities.size();

        check_num_threads(use_process_mask, t, num_threads, ec);

        if (use_process_mask)
        {
            used_cores = 0;
            max_cores = t.get_number_of_cores();
        }

        std::size_t num_cores = (std::min)(max_cores, t.get_number_of_cores());

        std::vector<std::size_t> num_pus_cores(num_cores, 0);
        std::vector<std::size_t> next_pu_index(num_cores, 0);
        std::vector<std::vector<std::size_t>> pu_indexes(num_cores);
        num_pus.resize(num_threads);

        // At first, calculate the number of used pus per core.
        // This needs to be done to make sure that we occupy all the available
        // cores
        for (std::size_t num_thread = 0; num_thread < num_threads; /**/)
        {
            for (std::size_t num_core = 0; num_core < num_cores; ++num_core)
            {
                std::size_t num_core_pus = t.get_number_of_core_pus(num_core);
                std::size_t pu_index = next_pu_index[num_core];
                bool use_pu = false;

                // Find the next PU on this core which is in the process mask
                while (pu_index < num_core_pus)
                {
                    use_pu = pu_in_process_mask(
                        use_process_mask, t, num_core, pu_index);
                    ++pu_index;

                    if (use_pu)
                    {
                        break;
                    }
                }

                next_pu_index[num_core] = pu_index;

                if (!use_pu)
                {
                    continue;
                }

                pu_indexes[num_core].push_back(next_pu_index[num_core] - 1);

                num_pus_cores[num_core]++;
                if (++num_thread == num_threads)
                    break;
            }
        }

        // Iterate over the cores and assigned pus per core. this additional
        // loop is needed so that we have consecutive worker thread numbers
        std::size_t num_thread = 0;
        for (std::size_t num_core = 0; num_core < num_cores; ++num_core)
        {
            for (std::size_t num_pu = 0; num_pu < num_pus_cores[num_core];
                 ++num_pu)
            {
                if (any(affinities[num_thread]))
                {
                    HPX_THROWS_IF(ec, bad_parameter,
                        "decode_balanced_distribution",
                        "affinity mask for thread {1} has already been set",
                        num_thread);
                    return;
                }

                num_pus[num_thread] = t.get_pu_number(
                    num_core + used_cores, pu_indexes[num_core][num_pu]);
                affinities[num_thread] = t.init_thread_affinity_mask(
                    num_core + used_cores, pu_indexes[num_core][num_pu]);
                ++num_thread;
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    void decode_numabalanced_distribution(topology& t,
        std::vector<mask_type>& affinities, std::size_t used_cores,
        std::size_t max_cores, std::vector<std::size_t>& num_pus,
        bool use_process_mask, error_code& ec)
    {
        HPX_UNUSED(max_cores);
        std::size_t num_threads = affinities.size();

        check_num_threads(use_process_mask, t, num_threads, ec);

        if (use_process_mask)
        {
            used_cores = 0;
        }

        num_pus.resize(num_threads);

        // numa nodes
        std::size_t num_numas =
            (std::max)(std::size_t(1), t.get_number_of_numa_nodes());
        std::vector<std::size_t> num_cores_numa(num_numas, 0);
        std::vector<std::size_t> num_pus_numa(num_numas, 0);
        std::vector<std::size_t> num_threads_numa(num_numas, 0);
        for (std::size_t n = 0; n < num_numas; ++n)
        {
            num_cores_numa[n] = t.get_number_of_numa_node_cores(n);
        }

        std::size_t core_offset = 0;
        std::size_t pus_t = 0;
        for (std::size_t n = 0; n < num_numas; ++n)
        {
            for (std::size_t num_core = 0; num_core < num_cores_numa[n];
                 ++num_core)
            {
                std::size_t num_pus =
                    t.get_number_of_core_pus(num_core + core_offset);
                for (std::size_t num_pu = 0; num_pu < num_pus; ++num_pu)
                {
                    if (pu_in_process_mask(use_process_mask, t,
                            num_core + core_offset, num_pu))
                    {
                        ++num_pus_numa[n];
                    }
                }
            }

            pus_t += num_pus_numa[n];
            core_offset += num_cores_numa[n];
        }

        // how many threads should go on each domain
        std::size_t pus_t2 = 0;
        for (std::size_t n = 0; n < num_numas; ++n)
        {
            std::size_t temp = static_cast<std::size_t>(
                std::round(static_cast<double>(num_threads * num_pus_numa[n]) /
                    static_cast<double>(pus_t)));

            // due to rounding up, we might have too many threads
            if ((pus_t2 + temp) > num_threads)
                temp = num_threads - pus_t2;
            pus_t2 += temp;
            num_threads_numa[n] = temp;

            // HPX_ASSERT(num_threads_numa[n] <= num_pus_numa[n]);
        }

        // HPX_ASSERT(num_threads <= pus_t2);

        // assign threads to cores on each numa domain
        std::size_t num_thread = 0;
        core_offset = 0;
        for (std::size_t n = 0; n < num_numas; ++n)
        {
            std::vector<std::size_t> num_pus_cores(num_cores_numa[n], 0);
            std::vector<std::size_t> next_pu_index(num_cores_numa[n], 0);
            std::vector<std::vector<std::size_t>> pu_indexes(num_cores_numa[n]);

            // iterate once and count pus/core
            for (std::size_t num_thread_numa = 0;
                 num_thread_numa < num_threads_numa[n];
                /**/)
            {
                for (std::size_t num_core = 0; num_core < num_cores_numa[n];
                     ++num_core)
                {
                    std::size_t num_core_pus =
                        t.get_number_of_core_pus(num_core);
                    std::size_t pu_index = next_pu_index[num_core];
                    bool use_pu = false;

                    // Find the next PU on this core which is in the process mask
                    while (pu_index < num_core_pus)
                    {
                        use_pu = pu_in_process_mask(use_process_mask, t,
                            num_core + core_offset, pu_index);
                        ++pu_index;

                        if (use_pu)
                        {
                            break;
                        }
                    }

                    next_pu_index[num_core] = pu_index;

                    if (!use_pu)
                    {
                        continue;
                    }

                    pu_indexes[num_core].push_back(next_pu_index[num_core] - 1);

                    num_pus_cores[num_core]++;
                    if (++num_thread_numa == num_threads_numa[n])
                        break;
                }
            }

            // Iterate over the cores and assigned pus per core. this additional
            // loop is needed so that we have consecutive worker thread numbers
            for (std::size_t num_core = 0; num_core < num_cores_numa[n];
                 ++num_core)
            {
                for (std::size_t num_pu = 0; num_pu < num_pus_cores[num_core];
                     ++num_pu)
                {
                    if (any(affinities[num_thread]))
                    {
                        HPX_THROWS_IF(ec, bad_parameter,
                            "decode_numabalanced_distribution",
                            "affinity mask for thread {1} has already been set",
                            num_thread);
                        return;
                    }
                    num_pus[num_thread] = t.get_pu_number(
                        num_core + used_cores, pu_indexes[num_core][num_pu]);
                    affinities[num_thread] = t.init_thread_affinity_mask(
                        num_core + used_cores + core_offset,
                        pu_indexes[num_core][num_pu]);
                    ++num_thread;
                }
            }
            core_offset += num_cores_numa[n];
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    void decode_distribution(distribution_type d, topology& t,
        std::vector<mask_type>& affinities, std::size_t used_cores,
        std::size_t max_cores, std::size_t num_threads,
        std::vector<std::size_t>& num_pus, bool use_process_mask,
        error_code& ec)
    {
        affinities.resize(num_threads);
        switch (d)
        {
        case compact:
            decode_compact_distribution(t, affinities, used_cores, max_cores,
                num_pus, use_process_mask, ec);
            break;

        case scatter:
            decode_scatter_distribution(t, affinities, used_cores, max_cores,
                num_pus, use_process_mask, ec);
            break;

        case balanced:
            decode_balanced_distribution(t, affinities, used_cores, max_cores,
                num_pus, use_process_mask, ec);
            break;

        case numa_balanced:
            decode_numabalanced_distribution(t, affinities, used_cores,
                max_cores, num_pus, use_process_mask, ec);
            break;

        default:
            HPX_ASSERT(false);
        }
    }
}}}    // namespace hpx::threads::detail

namespace hpx { namespace threads {
    ///////////////////////////////////////////////////////////////////////////
    void parse_affinity_options(std::string const& spec,
        std::vector<mask_type>& affinities, std::size_t used_cores,
        std::size_t max_cores, std::size_t num_threads,
        std::vector<std::size_t>& num_pus, bool use_process_mask,
        error_code& ec)
    {
        detail::mappings_type mappings;
        detail::parse_mappings(spec, mappings, ec);
        if (ec)
            return;

        // We need to instantiate a new topology object as the runtime has not
        // been initialized yet
        threads::topology& t = threads::create_topology();

        switch (mappings.which())
        {
        case 0:
        {
            detail::decode_distribution(
                boost::get<detail::distribution_type>(mappings), t, affinities,
                used_cores, max_cores, num_threads, num_pus, use_process_mask,
                ec);
            if (ec)
                return;
        }
        break;

        case 1:
        {
            if (use_process_mask)
            {
                HPX_THROWS_IF(ec, bad_parameter, "parse_affinity_options",
                    "can't use --hpx:use-process-mask with custom thread "
                    "bindings");
            }
            detail::mappings_spec_type mappings_specs(
                boost::get<detail::mappings_spec_type>(mappings));

            affinities.clear();
            for (detail::full_mapping_type& m : mappings_specs)
            {
                if (m.first.type_ != detail::spec_type::thread)
                {
                    HPX_THROWS_IF(ec, bad_parameter, "parse_affinity_options",
                        "bind specification ({1}) is ill formatted", spec);
                    return;
                }

                if (m.second.size() != 3)
                {
                    HPX_THROWS_IF(ec, bad_parameter, "parse_affinity_options",
                        "bind specification ({1}) is ill formatted", spec);
                    return;
                }

                if (m.second[0].type_ == detail::spec_type::unknown &&
                    m.second[1].type_ == detail::spec_type::unknown &&
                    m.second[2].type_ == detail::spec_type::unknown)
                {
                    HPX_THROWS_IF(ec, bad_parameter, "parse_affinity_options",
                        "bind specification ({1}) is ill formatted", spec);
                    return;
                }

                // repeat for each of the threads in the affinity specification
                detail::bounds_type thread_bounds =
                    extract_bounds(m.first, num_threads, ec);
                if (ec)
                    return;

                mappings_sanity_checks(m, num_threads, thread_bounds, ec);
                if (ec)
                    return;

                detail::decode_mappings(
                    t, m, affinities, thread_bounds.size(), ec);
                if (ec)
                    return;
            }

            if (num_pus.empty())
            {
                num_pus.resize(affinities.size());
                for (std::size_t i = 0; i != affinities.size(); ++i)
                {
                    num_pus[i] = threads::find_first(affinities[i]);
                }
            }
        }
        break;
        }
    }
}}    // namespace hpx::threads
