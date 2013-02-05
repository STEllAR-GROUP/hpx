//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//  Copyright (c) 2012-2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

#if defined(HPX_HAVE_HWLOC)

#include <hpx/exception.hpp>
#include <hpx/util/spinlock.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/runtime/threads/policies/hwloc_topology.hpp>

#include <hwloc.h>

#include <boost/format.hpp>

// #define BOOST_SPIRIT_DEBUG
#define BOOST_SPIRIT_USE_PHOENIX_V3
#include <boost/spirit/include/qi_char.hpp>
#include <boost/spirit/include/qi_nonterminal.hpp>
#include <boost/spirit/include/qi_numeric.hpp>
#include <boost/spirit/include/qi_operator.hpp>
#include <boost/spirit/include/qi_parse.hpp>
#include <boost/spirit/include/qi_string.hpp>
#include <boost/spirit/include/qi_auxiliary.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/include/std_pair.hpp>

#include <hpx/runtime/threads/detail/partlit.hpp>

namespace hpx { namespace threads
{
    hwloc_topology::hwloc_topology()
      : topo(0), machine_affinity_mask_(0)
    { // {{{
        int err = hwloc_topology_init(&topo);
        if (err != 0)
        {
            HPX_THROW_EXCEPTION(no_success, "hwloc_topology::hwloc_topology",
                "Failed to init hwloc topology");
        }

        err = hwloc_topology_load(topo);
        if (err != 0)
        {
            HPX_THROW_EXCEPTION(no_success, "hwloc_topology::hwloc_topology",
                "Failed to load hwloc topology");
        }

        init_num_of_pus();

        socket_numbers_.reserve(num_of_pus_);
        numa_node_numbers_.reserve(num_of_pus_);
        core_numbers_.reserve(num_of_pus_);

        // Initialize each set of data entirely, as some of the initialization
        // routines rely on access to other pieces of topology data. The
        // compiler will optimize the loops where possible anyways.

        std::size_t num_of_sockets = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_SOCKET);
        if(num_of_sockets == 0) num_of_sockets = 1;
        for (std::size_t i = 0; i < num_of_pus_; ++i)
        {
            std::size_t socket = init_socket_number(i);
            BOOST_ASSERT(socket < num_of_sockets);
            socket_numbers_.push_back(socket);
        }

        std::size_t num_of_nodes = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_NODE);
        if(num_of_nodes == 0) num_of_nodes = 1;
        for (std::size_t i = 0; i < num_of_pus_; ++i)
        {
            std::size_t numa_node = init_numa_node_number(i);
            BOOST_ASSERT(numa_node < num_of_nodes);
            numa_node_numbers_.push_back(numa_node);
        }

        std::size_t num_of_cores = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_CORE);
        if(num_of_cores == 0) num_of_cores = 1;
        for (std::size_t i = 0; i < num_of_pus_; ++i)
        {
            std::size_t core_number = init_core_number(i);
            BOOST_ASSERT(core_number < num_of_cores);
            core_numbers_.push_back(core_number);
        }

        machine_affinity_mask_ = init_machine_affinity_mask();
        socket_affinity_masks_.reserve(num_of_pus_);
        numa_node_affinity_masks_.reserve(num_of_pus_);
        core_affinity_masks_.reserve(num_of_pus_);
        thread_affinity_masks_.reserve(num_of_pus_);

        for (std::size_t i = 0; i < num_of_pus_; ++i)
        {
            socket_affinity_masks_.push_back(init_socket_affinity_mask(i));
        }

        for (std::size_t i = 0; i < num_of_pus_; ++i)
        {
            numa_node_affinity_masks_.push_back(init_numa_node_affinity_mask(i));
        }

        for (std::size_t i = 0; i < num_of_pus_; ++i)
        {
            core_affinity_masks_.push_back(init_core_affinity_mask(i));
        }

        for (std::size_t i = 0; i < num_of_pus_; ++i)
        {
            thread_affinity_masks_.push_back(init_thread_affinity_mask(i));
        }
    } // }}}

    hwloc_topology::~hwloc_topology()
    {
        if (topo)
            hwloc_topology_destroy(topo);
    }

    std::size_t hwloc_topology::get_socket_number(
        std::size_t num_thread
      , error_code& ec
        ) const
    { // {{{
        std::size_t num_pu = num_thread % num_of_pus_;

        if (num_pu < socket_numbers_.size())
        {
            if (&ec != &throws)
                ec = make_success_code();

            return socket_numbers_[num_pu];
        }

        HPX_THROWS_IF(ec, bad_parameter
          , "hpx::threads::hwloc_topology::get_socket_number"
          , boost::str(boost::format(
                "thread number %1% is out of range")
                % num_thread));
        return std::size_t(-1);
    } // }}}

    std::size_t hwloc_topology::get_numa_node_number(
        std::size_t num_thread
      , error_code& ec
        ) const
    { // {{{
        std::size_t num_pu = num_thread % num_of_pus_;

        if (num_pu < numa_node_numbers_.size())
        {
            if (&ec != &throws)
                ec = make_success_code();

            return numa_node_numbers_[num_pu];
        }

        HPX_THROWS_IF(ec, bad_parameter
          , "hpx::threads::hwloc_topology::get_numa_node_number"
          , boost::str(boost::format(
                "thread number %1% is out of range")
                % num_thread));
        return std::size_t(-1);
    } // }}}

    std::size_t hwloc_topology::get_core_number(
        std::size_t num_thread
      , error_code& ec
        ) const
    { // {{{
        std::size_t num_pu = num_thread % num_of_pus_;

        if (num_pu < core_numbers_.size())
        {
            if (&ec != &throws)
                ec = make_success_code();

            return core_numbers_[num_pu];
        }

        HPX_THROWS_IF(ec, bad_parameter
          , "hpx::threads::hwloc_topology::get_core_number"
          , boost::str(boost::format(
                "thread number %1% is out of range")
                % num_thread));
        return std::size_t(-1);
    } // }}}

    ///////////////////////////////////////////////////////////////////////////
    mask_type hwloc_topology::get_machine_affinity_mask(
        error_code& ec
        ) const
    {
        if (&ec != &throws)
            ec = make_success_code();

        return machine_affinity_mask_;
    }

    mask_type hwloc_topology::get_socket_affinity_mask(
        std::size_t num_thread
      , bool numa_sensitive
      , error_code& ec
        ) const
    { // {{{
        std::size_t num_pu = num_thread % num_of_pus_;

        if (num_pu < socket_affinity_masks_.size())
        {
            if (&ec != &throws)
                ec = make_success_code();

            return socket_affinity_masks_[num_pu];
        }

        HPX_THROWS_IF(ec, bad_parameter
          , "hpx::threads::hwloc_topology::get_socket_affinity_mask"
          , boost::str(boost::format(
                "thread number %1% is out of range")
                % num_thread));
        return 0;
    } // }}}

    mask_type hwloc_topology::get_numa_node_affinity_mask(
        std::size_t num_thread
      , bool numa_sensitive
      , error_code& ec
        ) const
    { // {{{
        std::size_t num_pu = num_thread % num_of_pus_;

        if (num_pu < numa_node_affinity_masks_.size())
        {
            if (&ec != &throws)
                ec = make_success_code();

            return numa_node_affinity_masks_[num_pu];
        }

        HPX_THROWS_IF(ec, bad_parameter
          , "hpx::threads::hwloc_topology::get_numa_node_affinity_mask"
          , boost::str(boost::format(
                "thread number %1% is out of range")
                % num_thread));
        return 0;
    } // }}}

    mask_type hwloc_topology::get_core_affinity_mask(
        std::size_t num_thread
      , bool numa_sensitive
      , error_code& ec
        ) const
    {
        std::size_t num_pu = num_thread % num_of_pus_;

        if (num_pu < core_affinity_masks_.size())
        {
            if (&ec != &throws)
                ec = make_success_code();

            return core_affinity_masks_[num_pu];
        }

        HPX_THROWS_IF(ec, bad_parameter
          , "hpx::threads::hwloc_topology::get_core_affinity_mask"
          , boost::str(boost::format(
                "thread number %1% is out of range")
                % num_thread));
        return 0;
    }

    mask_type hwloc_topology::get_thread_affinity_mask(
        std::size_t num_thread
      , bool numa_sensitive
      , error_code& ec
        ) const
    { // {{{
        std::size_t num_pu = num_thread % num_of_pus_;

        if (num_pu < thread_affinity_masks_.size())
        {
            if (&ec != &throws)
                ec = make_success_code();

            return thread_affinity_masks_[num_pu];
        }

        HPX_THROWS_IF(ec, bad_parameter
          , "hpx::threads::hwloc_topology::get_thread_affinity_mask"
          , boost::str(boost::format(
                "thread number %1% is out of range")
                % num_thread));
        return 0;
    } // }}}

    ///////////////////////////////////////////////////////////////////////////
    void hwloc_topology::set_thread_affinity_mask(
        boost::thread&
      , mask_type //mask
      , error_code& ec
        ) const
    {
        if (&ec != &throws)
            ec = make_success_code();
    }

    void hwloc_topology::set_thread_affinity_mask(
        mask_type mask
      , error_code& ec
        ) const
    { // {{{
        hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();

        for (std::size_t i = 0; i < sizeof(std::size_t) * CHAR_BIT; ++i)
        {
            if (mask & (static_cast<std::size_t>(1) << i))
            {
                hwloc_bitmap_set(cpuset, static_cast<unsigned int>(i));
            }
        }

        {
            scoped_lock lk(topo_mtx);
            if (hwloc_set_cpubind(topo, cpuset,
                  HWLOC_CPUBIND_STRICT | HWLOC_CPUBIND_THREAD))
            {
                // Strict binding not supported or failed, try weak binding.
                if (hwloc_set_cpubind(topo, cpuset, HWLOC_CPUBIND_THREAD))
                {
                    hwloc_bitmap_free(cpuset);

                    HPX_THROWS_IF(ec, kernel_error
                      , "hpx::threads::hwloc_topology::set_thread_affinity_mask"
                      , boost::str(boost::format(
                            "failed to set thread %x affinity mask")
                            % mask));

                    if (ec)
                        return;
                }
            }
        }
#if defined(__linux) || defined(linux) || defined(__linux__) || defined(__FreeBSD__)
        sleep(0);   // Allow the OS to pick up the change.
#endif

        hwloc_bitmap_free(cpuset);

        if (&ec != &throws)
            ec = make_success_code();
    } // }}}

    ///////////////////////////////////////////////////////////////////////////
    mask_type hwloc_topology::get_thread_affinity_mask_from_lva(
        naming::address::address_type
      , error_code& ec
        ) const
    { // {{{
        if (&ec != &throws)
            ec = make_success_code();

        return 0;
    } // }}}

    std::size_t hwloc_topology::init_node_number(
        std::size_t num_thread, hwloc_obj_type_t type
        )
    { // {{{
        if (std::size_t(-1) == num_thread)
            return std::size_t(-1);

        std::size_t num_pu = num_thread % num_of_pus_;

        {
            hwloc_obj_t obj;

            {
                scoped_lock lk(topo_mtx);
                obj = hwloc_get_obj_by_type(topo, HWLOC_OBJ_PU,
                    static_cast<unsigned>(num_pu));
            }

            while (obj)
            {
                if (hwloc_compare_types(obj->type, type) == 0)
                {
                    if (obj->os_index != ~0x0u)
                        return static_cast<std::size_t>(obj->os_index);

                    // on Windows os_index is always -1
                    return static_cast<std::size_t>(obj->logical_index);
                }
                obj = obj->parent;
            }
        }
        return 0;
    } // }}}

    void hwloc_topology::extract_node_mask(
        hwloc_obj_t parent
      , mask_type& mask
        ) const
    { // {{{
        hwloc_obj_t obj;

        {
            scoped_lock lk(topo_mtx);
            obj = hwloc_get_next_child(topo, parent, NULL);
        }

        while (obj)
        {
            if (hwloc_compare_types(HWLOC_OBJ_PU, obj->type) == 0)
            {
                do {
                    mask |= (static_cast<mask_type>(1) << obj->os_index);
                    {
                        scoped_lock lk(topo_mtx);
                        obj = hwloc_get_next_child(topo, parent, obj);
                    }
                } while (obj != NULL &&
                         hwloc_compare_types(HWLOC_OBJ_PU, obj->type) == 0);
                return;
            }

            extract_node_mask(obj, mask);

            scoped_lock lk(topo_mtx);
            obj = hwloc_get_next_child(topo, parent, obj);
        }
    } // }}}

    mask_type hwloc_topology::init_machine_affinity_mask() const
    { // {{{
        mask_type machine_affinity_mask = 0;

        hwloc_obj_t machine_obj;
        {
            scoped_lock lk(topo_mtx);
            machine_obj = hwloc_get_obj_by_type(topo, HWLOC_OBJ_MACHINE, 0);
        }
        if (machine_obj)
        {
            extract_node_mask(machine_obj, machine_affinity_mask);
            return machine_affinity_mask;
        }

        HPX_THROW_EXCEPTION(kernel_error
          , "hpx::threads::hwloc_topology::init_machine_affinity_mask"
          , "failed to initialize machine affinity mask");
        return 0;
    } // }}}

    mask_type hwloc_topology::init_socket_affinity_mask_from_socket(
        std::size_t num_socket
        ) const
    { // {{{
        // If we have only one or no socket, the socket affinity mask
        // spans all processors
        if (std::size_t(-1) == num_socket)
            return machine_affinity_mask_;

        hwloc_obj_t socket_obj;

        {
            scoped_lock lk(topo_mtx);
            socket_obj = hwloc_get_obj_by_type(topo,
                HWLOC_OBJ_SOCKET, static_cast<unsigned>(num_socket));
        }

        if (socket_obj)
        {
            mask_type socket_affinity_mask = 0;
            extract_node_mask(socket_obj, socket_affinity_mask);
            return socket_affinity_mask;
        }

        return machine_affinity_mask_;
    } // }}}

    mask_type hwloc_topology::init_numa_node_affinity_mask_from_numa_node(
        std::size_t numa_node
        ) const
    { // {{{
        // If we have only one or no NUMA domain, the NUMA affinity mask
        // spans all processors
        if (std::size_t(-1) == numa_node)
        {
            return machine_affinity_mask_;
        }

        hwloc_obj_t numa_node_obj;

        {
            scoped_lock lk(topo_mtx);
            numa_node_obj = hwloc_get_obj_by_type(topo,
                HWLOC_OBJ_NODE, static_cast<unsigned>(numa_node));
        }

        if (numa_node_obj)
        {
            mask_type node_affinity_mask = 0;
            extract_node_mask(numa_node_obj, node_affinity_mask);
            return node_affinity_mask;
        }

        return machine_affinity_mask_;
    } // }}}

    mask_type hwloc_topology::init_core_affinity_mask_from_core(
        std::size_t core, mask_type default_mask
        ) const
    { // {{{
        if (std::size_t(-1) == core)
            return default_mask;

        hwloc_obj_t core_obj;

        {
            scoped_lock lk(topo_mtx);
            core_obj = hwloc_get_obj_by_type(topo,
                HWLOC_OBJ_CORE, static_cast<unsigned>(core));
        }

        if (core_obj)
        {
            mask_type core_affinity_mask = 0;
            extract_node_mask(core_obj, core_affinity_mask);
            return core_affinity_mask;
        }

        return default_mask;
    } // }}}

    mask_type hwloc_topology::init_thread_affinity_mask(
        std::size_t num_thread
        ) const
    { // {{{

        if (std::size_t(-1) == num_thread)
        {
            return get_core_affinity_mask(num_thread, false);
        }

        std::size_t num_pu = num_thread % num_of_pus_;

        hwloc_obj_t obj;

        {
            scoped_lock lk(topo_mtx);
            obj = hwloc_get_obj_by_type(topo, HWLOC_OBJ_PU,
                    static_cast<unsigned>(num_pu));
        }

        if (!obj)
        {
            return get_core_affinity_mask(num_thread, false);
        }

        mask_type mask = 0x0u;

        mask |= (static_cast<mask_type>(1) << obj->os_index);

        return mask;
    } // }}}

    ///////////////////////////////////////////////////////////////////////////
    void hwloc_topology::init_num_of_pus()
    {
        num_of_pus_ = 1;
        {
            scoped_lock lk(topo_mtx);
            int num_of_pus = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_PU);

            if (num_of_pus > 0)
                num_of_pus_ = static_cast<std::size_t>(num_of_pus);
        }
    }
}}

///////////////////////////////////////////////////////////////////////////////
BOOST_FUSION_ADAPT_STRUCT(
    hpx::threads::detail::spec_type,
    (hpx::threads::mask_type, index_min_)
    (hpx::threads::mask_type, index_max_)
    (hpx::threads::detail::spec_type::type, type_)
)

namespace hpx { namespace threads { namespace detail
{
    static char const* const type_names[] =
    {
        "unknown", "thread", "socket", "numanode", "core", "pu"
    };

    char const* const spec_type::type_name(spec_type::type t)
    {
        if (t < spec_type::unknown || t > spec_type::pu)
            return type_names[0];
        return type_names[t];
    }

    ///////////////////////////////////////////////////////////////////////////
    //
    //    mappings:
    //        mapping(;mapping)*
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
    //        type:int
    //        type:int-int
    //        type:all
    //        ~pu-spec
    //
    //    type:
    //        socket | numanode
    //        core
    //        pu
    //
    namespace qi = boost::spirit::qi;

    // parser for affinity options
    template <typename Iterator>
    struct mappings_parser : qi::grammar<Iterator, mappings_type()>
    {
        mappings_parser()
          : mappings_parser::base_type(start)
        {
            using detail::partlit;
            using detail::spec_type;

            start = mapping % ';';

            mapping =  thread_spec >> '=' >> pu_spec;

            thread_spec =
                    partlit("thread") >> ':'
                    >>  (   qi::uint_
                            >>  ('-' >> qi::uint_ | qi::attr(0ul))
                            >>  qi::attr(spec_type::thread)
                        |   partlit("all") >> qi::attr(~0x0ul) >> qi::attr(0ul)
                            >>  qi::attr(spec_type::thread)
                        )
                ;

            pu_spec =
                    socket_spec >> core_spec >> processing_unit_spec
//                 |   '~' >> pu_spec
                ;

            socket_spec =
                    partlit("socket") >> ':'
                    >>  (   qi::uint_
                            >>  ('-' >> qi::uint_ | qi::attr(0ul))
                            >>  qi::attr(spec_type::socket)
                        |   partlit("all") >> qi::attr(~0x0ul) >> qi::attr(0ul)
                            >>  qi::attr(spec_type::socket)
                        )
                |   partlit("numanode") >> ':'
                    >>  (   qi::uint_
                            >>  ('-' >> qi::uint_ | qi::attr(0ul))
                            >>  qi::attr(spec_type::numanode)
                        |   partlit("all") >> qi::attr(~0x0ul) >> qi::attr(0ul)
                            >>  qi::attr(spec_type::numanode)
                        )
                |   qi::attr(0ul) >> qi::attr(0ul) >> qi::attr(spec_type::unknown)
                ;

            core_spec =
                    -qi::lit('.') >> partlit("core") >> ':'
                    >>  (   qi::uint_
                            >>  ('-' >> qi::uint_ | qi::attr(0ul))
                            >>  qi::attr(spec_type::core)
                        |   partlit("all") >> qi::attr(~0x0ul) >> qi::attr(0ul)
                            >>  qi::attr(spec_type::core)
                        )
                |   qi::attr(0ul) >> qi::attr(0ul) >> qi::attr(spec_type::unknown)
                ;

            processing_unit_spec =
                    -qi::lit('.') >> partlit("pu") >> ':'
                    >>  (   qi::uint_
                            >>  ('-' >> qi::uint_ | qi::attr(0ul))
                            >>  qi::attr(spec_type::pu)
                        |   partlit("all") >> qi::attr(~0x0ul) >> qi::attr(0ul)
                            >>  qi::attr(spec_type::pu)
                        )
                |   qi::attr(0ul) >> qi::attr(0ul) >> qi::attr(spec_type::unknown)
                ;

            BOOST_SPIRIT_DEBUG_NODES(
                (start)(mapping)(thread_spec)(pu_spec)
                (socket_spec)(core_spec)(processing_unit_spec)
            );
        }

        qi::rule<Iterator, mappings_type()> start;
        qi::rule<Iterator, full_mapping_type()> mapping;
        qi::rule<Iterator, spec_type()> thread_spec;
        qi::rule<Iterator, mapping_type()> pu_spec;
        qi::rule<Iterator, spec_type()> socket_spec;
        qi::rule<Iterator, spec_type()> core_spec;
        qi::rule<Iterator, spec_type()> processing_unit_spec;
    };

    template <typename Iterator>
    inline bool parse(Iterator& begin, Iterator end, mappings_type& m)
    {
        mappings_parser<Iterator> p;
        return qi::parse(begin, end, p, m);
    }

    ///////////////////////////////////////////////////////////////////////////
    void parse_mappings(std::string const& spec, mappings_type& mappings,
        error_code& ec)
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
    std::pair<std::size_t, std::size_t> extract_bounds(spec_type const& m,
        std::size_t default_last, error_code& ec)
    {
        std::size_t first = m.index_min_;
        std::size_t last = m.index_max_;

        if (first == ~0x0ul) {
            if (last == ~0x0ul) {
                HPX_THROWS_IF(ec, bad_parameter, "decode_mappings",
                    boost::str(boost::format("invalid thread specification, "
                        "min: %x, max %x") % first % last));
                return std::make_pair(first, last);
            }

            // bind all entities
            first = 0;
            last = default_last;
        }
        if (0 == last)
            last = first;

        if (&ec != &throws)
            ec = make_success_code();

        return std::make_pair(first, last);
    }

    ///////////////////////////////////////////////////////////////////////////
    mask_type decode_mapping2_unknown(hwloc_topology const& t,
        mapping_type const& m, mask_type mask, std::size_t size,
        error_code& ec)
    {
        if (&ec != &throws)
            ec = make_success_code();
        return mask;
    }

    mask_type decode_mapping_pu(hwloc_topology const& t,
        mapping_type const& m, std::size_t size, mask_type mask,
        error_code& ec)
    {
        std::pair<std::size_t, std::size_t> b = extract_bounds(m[2], size, ec);
        if (ec) return 0;

        mask_type pu_mask = 0;
        for (std::size_t i = b.first; i <= b.second; ++i)
            pu_mask |= t.init_thread_affinity_mask(i);

        return mask & pu_mask;
    }

    ///////////////////////////////////////////////////////////////////////////
    mask_type decode_mapping1_unknown(hwloc_topology const& t,
        mapping_type const& m, std::size_t size, mask_type mask,
        error_code& ec)
    {
        switch (m[2].type_) {
        case spec_type::pu:
            return decode_mapping_pu(t, m, size, mask, ec);

        case spec_type::unknown:
            return decode_mapping2_unknown(t, m, size, mask, ec);

        default:
            HPX_THROWS_IF(ec, bad_parameter, "decode_mapping1_unknown",
                boost::str(boost::format("unexpected specification type at "
                    "index two: %x (%s)") % m[1].type_ %
                        spec_type::type_name(m[1].type_)));
            break;
        }
        return mask;
    }

    mask_type decode_mapping_core(hwloc_topology const& t,
        mapping_type const& m, std::size_t size, mask_type mask,
        error_code& ec)
    {
        std::pair<std::size_t, std::size_t> b = extract_bounds(m[1], size, ec);
        if (ec) return 0;

        mask_type core_mask = 0;
        for (std::size_t i = b.first; i <= b.second; ++i)
            core_mask |= t.init_core_affinity_mask_from_core(i, 0);

        return decode_mapping1_unknown(t, m, size, mask & core_mask, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    mask_type decode_mapping0_unknown(hwloc_topology const& t,
        mapping_type const& m, std::size_t size, mask_type mask,
        error_code& ec)
    {
        switch (m[1].type_) {
        case spec_type::core:
            return decode_mapping_core(t, m, size, mask, ec);

        case spec_type::unknown:
            return decode_mapping1_unknown(t, m, size, mask, ec);

        default:
            HPX_THROWS_IF(ec, bad_parameter, "decode_mapping0_unknown",
                boost::str(boost::format("unexpected specification type at "
                    "index one: %x (%s)") % m[1].type_ %
                        spec_type::type_name(m[1].type_)));
            break;
        }
        return mask;
    }

    mask_type decode_mapping_socket(hwloc_topology const& t,
        mapping_type const& m, std::size_t size, error_code& ec)
    {
        std::pair<std::size_t, std::size_t> b = extract_bounds(m[0], size, ec);
        if (ec) return 0;

        mask_type mask = 0;
        for (std::size_t i = b.first; i <= b.second; ++i)
            mask |= t.init_socket_affinity_mask_from_socket(i);

        return decode_mapping0_unknown(t, m, size, mask, ec);
    }

    mask_type decode_mapping_numanode(hwloc_topology const& t,
        mapping_type const& m, std::size_t size, error_code& ec)
    {
        std::pair<std::size_t, std::size_t> b = extract_bounds(m[0], size, ec);
        if (ec) return 0;

        mask_type mask = 0;
        for (std::size_t i = b.first; i <= b.second; ++i)
            mask |= t.init_numa_node_affinity_mask_from_numa_node(i);

        return decode_mapping0_unknown(t, m, size, mask, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    void decode_mapping(hwloc_topology const& t, std::size_t thread_num,
        mapping_type const& m, std::vector<mask_type>& affinities,
        error_code& ec)
    {
        if (m.size() != 3) {
            HPX_THROWS_IF(ec, bad_parameter, "decode_mapping",
                "bad size of mappings specification array");
            return;
        }

        std::size_t size = affinities.size();
        mask_type mask = 0;
        switch (m[0].type_) {
        case spec_type::socket:
            // requested top level is a socket
            mask = decode_mapping_socket(t, m, size, ec);
            break;

        case spec_type::numanode:
            // requested top level is a NUMA node
            mask = decode_mapping_numanode(t, m, size, ec);
            break;

        case spec_type::unknown:
            // no top level is requested
            mask = decode_mapping0_unknown(t, m, size,
                t.get_machine_affinity_mask(), ec);
            break;

        default:
            HPX_THROWS_IF(ec, bad_parameter, "decode_mapping",
                boost::str(boost::format("unexpected specification type at "
                    "index zero: %x (%s)") % m[0].type_ %
                        spec_type::type_name(m[0].type_)));
            return;
        }

        // set each thread affinity only once
        if (0 != affinities[thread_num])
        {
            HPX_THROWS_IF(ec, bad_parameter, "decode_mapping",
                boost::str(boost::format("affinity mask for thread %1% has "
                    "already been set") % thread_num));
            return;
        }

        // set result
        affinities[thread_num] = mask;
    }

    void decode_mappings(full_mapping_type const& m,
        std::vector<mask_type>& affinities, error_code& ec)
    {
        // We need to instantiate a new topology object as the runtime has not
        // been initialized yet
        hwloc_topology t;

        // repeat for each of the threads in the affinity specification
        std::pair<std::size_t, std::size_t> bounds =
            extract_bounds(m.first, affinities.size(), ec);
        if (ec) return;

        for (std::size_t i = bounds.first; i <= bounds.second; ++i)
        {
            decode_mapping(t, i, m.second, affinities, ec);
            if (ec) return;
        }
    }
}}}

namespace hpx { namespace threads
{
    ///////////////////////////////////////////////////////////////////////////
    void parse_affinity_options(std::string const& spec,
        std::vector<mask_type>& affinities, error_code& ec)
    {
        detail::mappings_type mappings;
        detail::parse_mappings(spec, mappings, ec);
        if (!ec) {
            BOOST_FOREACH(detail::full_mapping_type const& m, mappings)
            {
                detail::decode_mappings(m, affinities, ec);
                if (ec) return;
            }
        }
    }
}}

#endif
