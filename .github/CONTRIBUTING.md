<!-- Copyright (c) 2014 Hartmut Kaiser                                            -->
<!--                                                                              -->
<!-- Distributed under the Boost Software License, Version 1.0. (See accompanying -->
<!-- file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)        -->

This describes how you can contribute to [HPX](https://github.com/STEllAR-GROUP/hpx).
Great to have you here. There are a few ways you can help make HPX better!

# How to Get Involved in Developing HPX

This page describes how you can get yourself involved with the development of
HPX. Here are some easy things to do.

The easiest ways to get in contact with us are listed here:

* Mailing list: [hpx-users@stellar.cct.lsu.edu](email:hpx-users@stellar.cct.lsu.edu), [hpx-devel@stellar.cct.lsu.edu](email:hpx-devel@stellar.cct.lsu.edu)
* IRC channel:  #ste||ar on irc.freenode.net
* Blog:         [stellar.cct.lsu.edu](stellar.cct.lsu.edu)

The basic approach is to find something fun you want to fix, hack it up, and
send a `git diff` as a mail attachment to [hpx-devel@stellar.cct.lsu.edu](email:hpx-devel@stellar.cct.lsu.edu)
with a Subject prefixed with 'PATCH', as well as: "made available under the Boost
Software License V1" license statement. We also need a real name for the git
commit logs if you usually use an alias. Alternatively, you can create a pull
request from your HPX repository you cloned on Github (see below).

It should be easy!

If you create new files, please use our License Header:

    //  Copyright (c) <year> <your name>
    //
    //  Distributed under the Boost Software License, Version 1.0. (See accompanying
    //  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

Please avoid larger reformatting of the code for the time being (except for the
tasks listed below).

If the task is really quick and easy, 'just do it'. However, if you think it
will take you some time, and/or need partitioning (eg. some big, scalable
cleanup that many people can help out with), then please:

* add a comment to the ticket that you're starting work on it
* please provide updates each week or so, otherwise someone else may take the issue
* please take only one part of the task at a time.

If a task has an owner without an update in a week, feel free to notify them
that you're taking that on yourself, and of course if you realize you can't
complete a task - please update it in the
[ticket system](https://github.com/STEllAR-GROUP/hpx/issues).

Even if you are deeply skilled, please consider doing one little easy hack, to
get used to the process. After that, you are invited to move on up to the more
difficult tasks, leaving some of the easy tasks to others so they can get
involved and achieve change themselves. The quicker you move up the pile, the
more quickly you can be making large scale, user-visible changes and
improvements to HPX - of which these easy hacks are just the tip of a very
interesting iceberg.

Before we get to the list of possible tasks, here is some additional
information to get you started.

## Getting Started

### Get a login on Github [here](https://github.com/) and fork the HPX repository to your Github account.

All new and old bugs in HPX can be found in our
[ticket system](https://github.com/STEllAR-GROUP/hpx/issues). Especially with
new incoming bugs, it is helpful to test the bug on your own computer/operating
system and comment in the bug entry whether you can or cannot confirm the bug
and under what circumstances it affects you.

### Getting a build - if necessary

Some but not all tasks require you to have built HPX. Even if that is not
required, your feedback can be helpful to us - so - please try. The master
build instructions are [here](http://stellar.cct.lsu.edu/files/hpx_0.9.8/html/hpx/tutorial/getting_started.html)
with more stuff under development.

### Hacking help

If you need to search constructs in the code, there is a code search engine at
the [Ohloh HPX page](http://code.ohloh.net/project?pid=&ipid=309791) or simply
at the top of this page.

## General info

We use the [Boost coding standards](http://www.boost.org/development/requirements.html#Guidelines)
for our work on HPX

The short version of the guidelines:

* 80-character lines.
* Absolutely no tabs (use spaces instead of tabs).
* Because we use git, UNIX line endings.
* Identifiers are C++ STL style: no CamelCase. E.g. `my_class` instead of `MyClass`.
* Use expressive identifiers.
* Exceptions for error handling instead of C-style error codes.

There is a `.editorconfig` file in the HPX root directory which can be used
for almost any widely available editor. Please see
[their webpage](http://editorconfig.org) to download plugins for your favorite
editor.

A few additional ones:

* Use doxygen style comments to document API functions.
* Before writing a piece of utility code, see if there is something in
  `hpx::util`, Boost or the C++ standard library that can be used to save time.

# Community

Community is an important part of all we do.

* You can help us answer questions our users have by being around on IRC 
  (#ste||ar on irc.freenode.net) or by chiming in on the
  [users mailing list](email:hpx-users@stellar.cct.lsu.edu)
* You can help write blog posts (for [stellar.cct.lsu.edu](stellar.cct.lsu.edu))
  about things you're doing with HPX. We can give you access or help with
  posting things.
* Create an example of how to use HPX in the real world by building something
  or showing what others have built.
* Write about other people’s work based on HPX. Show how it’s used in daily
  life. Take screenshots and make videos!


# Your first bugfix

For our project, you can talk to the following people to receive help in
working through your first bugfix and thinking through the problem:

* @hkaiser, @heller, @wash

