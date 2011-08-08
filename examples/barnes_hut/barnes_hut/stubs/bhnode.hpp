////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2011 Daniel Kogler
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////
#ifndef _BARNES_HUT_OCTNODE_STUBS_HPP
#define _BARNES_HUT_OCTNODE_STUBS_HPP

/*This is the bhnode stub file.
*/

#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/lcos/eager_future.hpp>

#include "../server/bhnode.hpp"

namespace hpx { namespace components { namespace stubs
{
    struct bhnode : stub_base<server::bhnode>
    {
        //constructors
        static server::bhnode::constFuture construct_node(
            const id_type gid, const double dat[7], const bool root){
            std::vector<double> theData(dat,dat+7);
            return lcos::eager_future<server::bhnode::constNodeAction,
                int>(gid,gid,theData,root);
        }
        static server::bhnode::cnstFuture2 construct_node(const id_type gid,
            const id_type insertPoint, const vector<double> bounds){
            return lcos::eager_future<server::bhnode::cnstNodeAction2,
                int>(gid,gid,insertPoint,bounds);
        }

        //other functions
        static int set_boundaries(const id_type gid, const id_type parId,
            const double bounds[6]){
            std::vector<double> theBounds(bounds,bounds+6);
            return lcos::eager_future<server::bhnode::setBoundsAction,
                int>(gid,parId,theBounds).get();
        }
        static server::bhnode::inPntFuture insert_node(const id_type gid,
            const vector<double> nodep, const double nodem,
            const id_type nodeGid){
            return lcos::eager_future<server::bhnode::findInPntAction,
                vector<double> >(gid,nodep,nodem,nodeGid);
        }
        static server::bhnode::runFuture run(const id_type gid,
            const id_type controllerGid, const vector<double> info){
            return lcos::eager_future<server::bhnode::runSimAction,
                int>(gid,controllerGid,info);
        }
    };
}}}

#endif
