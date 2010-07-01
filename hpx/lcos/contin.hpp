//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_CONTIN_JUN_27_2008_0420PM)
#define HPX_LCOS_CONTIN_JUN_27_2008_0420PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/util/full_empty_memory.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/lcos/future_value.hpp>
#include <hpx/util/block_profiler.hpp>

#include <boost/variant.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos 
{
    ///////////////////////////////////////////////////////////////////////////
    /// \class contin contin.hpp hpx/lcos/contin.hpp
    ///
    template <typename Action, typename Result, typename DirectExecute>
    class contin;

    ///////////////////////////////////////////////////////////////////////////
    struct contin_tag {};

    template <typename Action, typename Result>
    class contin<Action, Result, boost::mpl::false_> 
        : public future_value<Result, typename Action::result_type>
    {
    private:
        typedef future_value<Result, typename Action::result_type> base_type;

    public:
        contin()
          : apply_logger_("contin::apply")
        {}

        naming::id_type gid()
        {
            return this->get_gid();
        }

        util::block_profiler<contin_tag> apply_logger_;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct contin_direct_tag {};

    template <typename Action, typename Result>
    class contin<Action, Result, boost::mpl::true_> 
        : public future_value<Result, typename Action::result_type>
    {
    private:
        typedef future_value<Result, typename Action::result_type> base_type;

    public:
        contin()
          : apply_logger_("contin_direct::apply")
        {}

        util::block_profiler<contin_direct_tag> apply_logger_;
    };

}}

#endif
