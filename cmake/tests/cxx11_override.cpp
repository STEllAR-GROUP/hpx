////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2017 Denis Blank
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

class my_base
{
  public:
    my_base() { }
    virtual ~my_base() { }

    virtual void do_sth() { }
};

class my_inh : public my_base
{
  public:
    my_inh() { }

    void do_sth() override { }
};

int main()
{
    my_base* obj = new my_inh();
    delete obj;
}
