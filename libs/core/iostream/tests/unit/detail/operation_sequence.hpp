//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Distributed under the Boost Software License, Version 1.0.(See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt.)
//
// See http://www.boost.org/libs/iostreams for documentation.
//
// Defines the classes operation_sequence and operation, in the namespace
// hpx::iostream::test, for verifying that all elements of a sequence of
// operations are executed, and that they are executed in the correct order.
//
// File:        libs/iostreams/test/detail/operation_sequence.hpp
// Date:        Mon Dec 10 18:58:19 MST 2007
// Copyright:   2007-2008 CodeRage, LLC
// Author:      Jonathan Turkanis
// Contact:     turkanis at coderage dot com

#pragma once

#include <climits>
#include <cstddef>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>    // pair
#include <vector>

namespace hpx::iostream::test {

    // Simple exception class with error code built in to type
    template <int Code>
    struct operation_error
    {
    };

    class operation_sequence;

    // Represent an operation in a sequence of operations to be executed
    class operation
    {
    public:
        friend class operation_sequence;

        operation() = default;

        operation(operation const&) = default;
        operation(operation&&) = default;
        operation& operator=(operation const&) = default;
        operation& operator=(operation&&) = default;

        ~operation() = default;

        void execute();

    private:
        static void remove_operation(operation_sequence& seq, int id);

        struct impl
        {
            impl(operation_sequence& seq, int id, int error_code = -1)
              : seq(seq)
              , id(id)
              , error_code(error_code)
            {
            }

            ~impl()
            {
                remove_operation(seq, id);
            }

            impl(impl const&) = delete;
            impl(impl&&) = delete;
            impl& operator=(impl const&) = delete;
            impl& operator=(impl&&) = delete;

            operation_sequence& seq;
            int id;
            int error_code;
        };
        friend struct impl;

        operation(operation_sequence& seq, int id, int error_code = -1)
          : pimpl_(std::make_shared<impl>(seq, id, error_code))
        {
        }

        std::shared_ptr<impl> pimpl_;
    };

    // Represents a sequence of operations to be executed in a particular order
    class operation_sequence
    {
    public:
        friend class operation;
        operation_sequence()
        {
            reset();
        }

        operation_sequence(operation_sequence const&) = delete;
        operation_sequence(operation_sequence&&) = delete;
        operation_sequence& operator=(operation_sequence const&) = delete;
        operation_sequence& operator=(operation_sequence&&) = delete;

        //
        // Returns a new operation.
        // Parameters:
        //
        //   id - The operation id, determining the position
        //        of the new operation in the operation sequence
        //   error_code - If supplied, indicates that the new
        //        operation will throw operation_error<error_code>
        //        when executed. Must be an integer between 0 and
        //        HPX_IOSTREAMS_TEST_MAX_OPERATION_ERROR,
        //        inclusive.
        //
        operation new_operation(int id, int error_code = INT_MAX);

        bool is_success() const
        {
            return success_;
        }

        bool is_failure() const
        {
            return failed_;
        }

        std::string message() const;
        void reset();

    private:
        void execute(int id);
        void remove_operation(int id);

        typedef std::weak_ptr<operation::impl> ptr_type;
        typedef std::map<int, ptr_type> map_type;

        map_type operations_;
        std::vector<int> log_;
        std::size_t total_executed_;
        int last_executed_;
        bool success_;
        bool failed_;
    };

    //--------------Implementation of operation-----------------------------------//
    void operation::execute()
    {
        pimpl_->seq.execute(pimpl_->id);
        switch (pimpl_->error_code)
        {
        case 0:
            throw operation_error<0>();
        case 1:
            throw operation_error<1>();
        case 2:
            throw operation_error<2>();
        case 3:
            throw operation_error<3>();
        case 4:
            throw operation_error<4>();
        case 5:
            throw operation_error<5>();
        case 6:
            throw operation_error<6>();
        case 7:
            throw operation_error<7>();
        case 8:
            throw operation_error<8>();
        case 9:
            throw operation_error<9>();
        case 10:
            throw operation_error<10>();
        case 11:
            throw operation_error<11>();
        case 12:
            throw operation_error<12>();
        case 13:
            throw operation_error<13>();
        case 14:
            throw operation_error<14>();
        case 15:
            throw operation_error<15>();
        case 16:
            throw operation_error<16>();
        case 17:
            throw operation_error<17>();
        case 18:
            throw operation_error<18>();
        case 19:
            throw operation_error<19>();
        default:
            break;
        }
    }

    inline void operation::remove_operation(operation_sequence& seq, int id)
    {
        seq.remove_operation(id);
    }

    //--------------Implementation of operation_sequence--------------------------//
    inline operation operation_sequence::new_operation(int id, int error_code)
    {
        using namespace std;
        if (error_code < 0)
        {
            throw runtime_error(string("The error code ") +
                std::to_string(error_code) + " is out of range");
        }

        if (last_executed_ != INT_MIN)
        {
            throw runtime_error("Operations in progress; call reset() "
                                "before creating more operations");
        }

        map_type::const_iterator it = operations_.find(id);
        if (it != operations_.end())
        {
            throw runtime_error(string("The operation ") + std::to_string(id) +
                " already exists");
        }

        operation op(*this, id, error_code);
        operations_.insert(make_pair(id, ptr_type(op.pimpl_)));
        return op;
    }

    inline std::string operation_sequence::message() const
    {
        using namespace std;
        if (success_)
            return "success";

        std::string msg = failed_ ? "operations occurred out of order: " :
                                    "operation sequence is incomplete: ";

        typedef vector<int>::size_type size_type;
        for (size_type z = 0, n = log_.size(); z < n; ++z)
        {
            msg += std::to_string(log_[z]);
            if (z < n - 1)
                msg += ',';
        }
        return msg;
    }

    inline void operation_sequence::reset()
    {
        log_.clear();
        total_executed_ = 0;
        last_executed_ = INT_MIN;
        success_ = false;
        failed_ = false;
    }

    inline void operation_sequence::execute(int id)
    {
        log_.push_back(id);
        if (!failed_ && last_executed_ < id)
        {
            if (++total_executed_ == operations_.size())
                success_ = true;
            last_executed_ = id;
        }
        else
        {
            success_ = false;
            failed_ = true;
        }
    }

    inline void operation_sequence::remove_operation(int id)
    {
        using namespace std;
        map_type::iterator it = operations_.find(id);
        if (it == operations_.end())
        {
            throw runtime_error(
                string("No such operation: ") + std::to_string(id));
        }
        operations_.erase(it);
    }
}    // namespace hpx::iostream::test
