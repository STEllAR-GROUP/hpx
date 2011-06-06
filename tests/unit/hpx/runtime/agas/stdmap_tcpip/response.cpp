//  Copyright (c) 2011 Bryce Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/agas/network/backend/tcpip.hpp>
#include <hpx/runtime/agas/namespace/response.hpp>
#include <hpx/util/lightweight_test.hpp>

typedef hpx::agas::response<
    hpx::agas::tag::network::tcpip
> response_type;

typedef hpx::agas::gva<
    hpx::agas::tag::network::tcpip
> gva_type;

using hpx::success;
using hpx::no_success;

using hpx::components::component_base_lco;
using hpx::components::component_memory_block;
using hpx::components::component_dataflow_block;

using hpx::naming::gid_type;

using hpx::agas::invalid_request;
using hpx::agas::primary_ns_bind_locality;
using hpx::agas::primary_ns_bind_gid;
using hpx::agas::primary_ns_resolve_locality;
using hpx::agas::primary_ns_resolve_gid;
using hpx::agas::primary_ns_unbind;
using hpx::agas::primary_ns_increment;
using hpx::agas::primary_ns_decrement;
using hpx::agas::primary_ns_localities;
using hpx::agas::component_ns_bind_prefix;
using hpx::agas::component_ns_bind_name;
using hpx::agas::component_ns_resolve_id;
using hpx::agas::component_ns_resolve_name;
using hpx::agas::component_ns_unbind;
using hpx::agas::symbol_ns_bind;
using hpx::agas::symbol_ns_rebind;
using hpx::agas::symbol_ns_resolve;
using hpx::agas::symbol_ns_unbind;

///////////////////////////////////////////////////////////////////////////////
int main(void)
{
    {   // default constructor
        response_type r;

        HPX_TEST_EQ(unsigned(invalid_request), unsigned(r.which()));
    }

    {   // primary_ns_bind_gid
        // component_ns_unbind
        // symbol_ns_bind
        // symbol_ns_unbind
        response_type r0(primary_ns_bind_gid, success)
                    , r1(primary_ns_bind_gid, no_success)
                    , r2(component_ns_unbind, success)
                    , r3(component_ns_unbind, no_success)
                    , r4(symbol_ns_bind,      success)
                    , r5(symbol_ns_bind,      no_success)
                    , r6(symbol_ns_unbind,    success)
                    , r7(symbol_ns_unbind,    no_success);

        HPX_TEST_EQ(unsigned(primary_ns_bind_gid), unsigned(r0.which()));
        HPX_TEST_EQ(unsigned(primary_ns_bind_gid), unsigned(r1.which()));
        HPX_TEST_EQ(unsigned(component_ns_unbind), unsigned(r2.which()));
        HPX_TEST_EQ(unsigned(component_ns_unbind), unsigned(r3.which()));
        HPX_TEST_EQ(unsigned(symbol_ns_bind),      unsigned(r4.which()));
        HPX_TEST_EQ(unsigned(symbol_ns_bind),      unsigned(r5.which()));
        HPX_TEST_EQ(unsigned(symbol_ns_unbind),    unsigned(r6.which()));
        HPX_TEST_EQ(unsigned(symbol_ns_unbind),    unsigned(r7.which()));

        HPX_TEST_EQ(unsigned(success),    unsigned(r0.get_status()));
        HPX_TEST_EQ(unsigned(no_success), unsigned(r1.get_status()));
        HPX_TEST_EQ(unsigned(success),    unsigned(r2.get_status()));
        HPX_TEST_EQ(unsigned(no_success), unsigned(r3.get_status()));
        HPX_TEST_EQ(unsigned(success),    unsigned(r4.get_status()));
        HPX_TEST_EQ(unsigned(no_success), unsigned(r5.get_status()));
        HPX_TEST_EQ(unsigned(success),    unsigned(r6.get_status()));
        HPX_TEST_EQ(unsigned(no_success), unsigned(r7.get_status()));
    }

    {   // component_ns_bind_prefix
        // component_ns_bind_name
        // component_ns_resolve_name
        response_type r0(component_ns_bind_prefix,  component_base_lco)
                    , r1(component_ns_bind_prefix,  56)
                    , r2(component_ns_bind_name,    component_memory_block)
                    , r3(component_ns_bind_name,    12)
                    , r4(component_ns_resolve_name, component_dataflow_block)
                    , r5(component_ns_resolve_name, 67);

        HPX_TEST_EQ(unsigned(component_ns_bind_prefix),  unsigned(r0.which()));
        HPX_TEST_EQ(unsigned(component_ns_bind_prefix),  unsigned(r1.which()));
        HPX_TEST_EQ(unsigned(component_ns_bind_name),    unsigned(r2.which()));
        HPX_TEST_EQ(unsigned(component_ns_bind_name),    unsigned(r3.which()));
        HPX_TEST_EQ(unsigned(component_ns_resolve_name), unsigned(r4.which()));
        HPX_TEST_EQ(unsigned(component_ns_resolve_name), unsigned(r5.which()));

        HPX_TEST_EQ(boost::int32_t(component_base_lco),
                    r0.get_component_type());
        HPX_TEST_EQ(boost::int32_t(56),
                    r1.get_component_type());
        HPX_TEST_EQ(boost::int32_t(component_memory_block),
                    r2.get_component_type());
        HPX_TEST_EQ(boost::int32_t(12),
                    r3.get_component_type());
        HPX_TEST_EQ(boost::int32_t(component_dataflow_block),
                    r4.get_component_type());
        HPX_TEST_EQ(boost::int32_t(67),
                    r5.get_component_type());
    }

    {   // symbol_ns_rebind
        // symbol_ns_resolve
        response_type r0(symbol_ns_rebind,  gid_type(5, 5))
                    , r1(symbol_ns_rebind,  gid_type(0xdead, 0xbeef))
                    , r2(symbol_ns_resolve, gid_type(17, 42))
                    , r3(symbol_ns_resolve, gid_type(60));

        HPX_TEST_EQ(unsigned(symbol_ns_rebind),  unsigned(r0.which()));
        HPX_TEST_EQ(unsigned(symbol_ns_rebind),  unsigned(r1.which()));
        HPX_TEST_EQ(unsigned(symbol_ns_resolve), unsigned(r2.which()));
        HPX_TEST_EQ(unsigned(symbol_ns_resolve), unsigned(r3.which()));

        HPX_TEST_EQ(gid_type(5, 5),           r0.get_gid());
        HPX_TEST_EQ(gid_type(0xdead, 0xbeef), r1.get_gid());
        HPX_TEST_EQ(gid_type(17, 42),         r2.get_gid());
        HPX_TEST_EQ(gid_type(60),             r3.get_gid());
    }

    {   // primary_ns_resolve_locality
        using boost::asio::ip::address;
        typedef boost::asio::ip::tcp::endpoint endpoint;

        endpoint ep0(address::from_string("192.168.2.9"), 80)
               , ep1(address::from_string("172.3.10.5"), 433);
       
        void* p0 = (void*) 0xdeadbeef;
        void* p1 = (void*) 0xcededeed;
 
        gva_type g0(ep0, component_base_lco, 0x100, p0, 0x10)
               , g1(ep1, component_memory_block, 0x20, p1, 0x2a); 

        response_type r0(primary_ns_resolve_locality, boost::uint32_t(1), g0)
                    , r1(primary_ns_resolve_locality, boost::uint32_t(2), g1);

        HPX_TEST_EQ(unsigned(primary_ns_resolve_locality),
                    unsigned(r0.which()));
        HPX_TEST_EQ(unsigned(primary_ns_resolve_locality),
                    unsigned(r1.which()));

        HPX_TEST_EQ(g0, r0.get_gva());
        HPX_TEST_EQ(g1, r1.get_gva());
    }

    
    {   // primary_ns_localities
        // component_ns_resolve_id
        boost::uint32_t* l0 = new boost::uint32_t [1];
        boost::uint32_t* l1 = new boost::uint32_t [10];
        boost::uint32_t* l2 = 0;
        boost::uint32_t* l3 = new boost::uint32_t [1];
        boost::uint32_t* l4 = new boost::uint32_t [16];
        boost::uint32_t* l5 = 0;

        l0[0] = 17;
        l3[0] = 42;

        for (boost::uint64_t i = 0; i < 10; ++i)
            l1[i] = 42 + (i * 2);

        for (boost::uint64_t i = 0; i < 16; ++i)
            l4[i] = i * i;

        response_type r0(primary_ns_localities,   boost::uint64_t(1),  l0)
                    , r1(primary_ns_localities,   boost::uint64_t(10), l1)
                    , r2(primary_ns_localities,   boost::uint64_t(0),  l2)
                    , r3(component_ns_resolve_id, boost::uint64_t(1),  l3)
                    , r4(component_ns_resolve_id, boost::uint64_t(16), l4)
                    , r5(component_ns_resolve_id, boost::uint64_t(0),  l5);

        HPX_TEST_EQ(unsigned(primary_ns_localities),   unsigned(r0.which()));
        HPX_TEST_EQ(unsigned(primary_ns_localities),   unsigned(r1.which()));
        HPX_TEST_EQ(unsigned(primary_ns_localities),   unsigned(r2.which()));
        HPX_TEST_EQ(unsigned(component_ns_resolve_id), unsigned(r3.which()));
        HPX_TEST_EQ(unsigned(component_ns_resolve_id), unsigned(r4.which()));
        HPX_TEST_EQ(unsigned(component_ns_resolve_id), unsigned(r5.which()));

        HPX_TEST_EQ(boost::uint64_t(1), r0.get_localities_size());
        HPX_TEST_EQ(17U, r0.get_localities()[0]);

        HPX_TEST_EQ(boost::uint64_t(10), r1.get_localities_size());
        for (boost::uint64_t i = 0; i < 10; ++i)
            HPX_TEST_EQ(boost::uint32_t(42 + (i * 2)), r1.get_localities()[i]);

        HPX_TEST_EQ(boost::uint64_t(0), r2.get_localities_size());
        HPX_TEST_EQ(l2, r2.get_localities());

        HPX_TEST_EQ(boost::uint64_t(1), r3.get_localities_size());
        HPX_TEST_EQ(42U, r3.get_localities()[0]);

        HPX_TEST_EQ(boost::uint64_t(16), r4.get_localities_size());
        for (boost::uint64_t i = 0; i < 16; ++i)
            HPX_TEST_EQ(boost::uint32_t(i * i), r4.get_localities()[i]);

        HPX_TEST_EQ(boost::uint64_t(0), r5.get_localities_size());
        HPX_TEST_EQ(l5, r5.get_localities());
    }

    return hpx::util::report_errors();
}

