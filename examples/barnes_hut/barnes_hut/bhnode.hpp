////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2011 Daniel Kogler
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////
#ifndef _BARNES_HUT_OCTNODE_HPP
#define _BARNES_HUT_OCTNODE_HPP

/*This is the bhnode interface header file.*/

#include <hpx/runtime.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include "stubs/bhnode.hpp"

namespace hpx { namespace components
{
    class bhnode : public client_base<bhnode, stubs::bhnode>
    {
        typedef client_base<bhnode, stubs::bhnode> base_type;

        public:
        //constructors 
        bhnode(){}
        bhnode(id_type gid) : base_type(gid){}

        //initialization functions
        server::bhnode::constFuture construct_node(const double dat[7],
            const bool root){
            BOOST_ASSERT(gid_);
            return this->base_type::construct_node(gid_,dat,root);
        }
        server::bhnode::cnstFuture2 construct_node(const id_type insertPoint,
            const vector<double> bounds){
            BOOST_ASSERT(gid_);
            return this->base_type::construct_node(gid_, insertPoint, bounds);
        }

        //other functions
        int set_boundaries(const id_type parId, const double bounds[6]){
            BOOST_ASSERT(gid_);
            return this->base_type::set_boundaries(gid_,parId,bounds);
        }

        server::bhnode::inPntFuture insert_node(const vector<double> nodep,
            const double nodem, const id_type nodeGid){
            BOOST_ASSERT(gid_);
            return this->base_type::insert_node(gid_,nodep,nodem,nodeGid);
        }

        server::bhnode::runFuture run(const id_type controllerGid,
            const vector<double> info){
            BOOST_ASSERT(gid_);
            return this->base_type::run(gid_,controllerGid,info);
        }
    };
}}

#endif
