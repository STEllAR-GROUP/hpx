////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <memory>

struct base
{};

struct derived
  : public base
  , std::enable_shared_from_this<derived>
{};


int main()
{
    std::shared_ptr<base> pb = std::make_shared<derived>();
    std::shared_ptr<derived> pd = std::static_pointer_cast<derived>(pb);
    std::shared_ptr<derived> sft = pd->shared_from_this();
}
