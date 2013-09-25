////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <memory>
#include <vector>

template<class T> using Alloc = std::allocator<T>;
template<class T> using Vec = std::vector<T, Alloc<T>>;

template<class T>
void process(Vec<T>& v)
{ /* ... */ }

template<template<class,class> class TT>
void g(TT<int, Alloc<int>>){}

int main()
{
    Vec<int> v; // same as vector<int, Alloc<int>> v;

    g(v); // OK: TT = vector
}
