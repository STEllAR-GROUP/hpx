# Copyright (c) 2018 Mikael Simberg
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

from docutils import nodes

def setup(app):
    app.add_role('hpx-issue', autolink('https://github.com/STEllAR-GROUP/hpx/issues/%s', "Issue #"))
    app.add_role('hpx-pr', autolink('https://github.com/STEllAR-GROUP/hpx/pull/%s', "PR #"))
    app.add_role('cppreference-header', autolink('http://en.cppreference.com/w/cpp/header/%s'))
    app.add_role('cppreference-algorithm', autolink('http://en.cppreference.com/w/cpp/algorithm/%s'))
    app.add_role('cppreference-memory', autolink('http://en.cppreference.com/w/cpp/memory/%s'))
    app.add_role('cppreference-container', autolink('http://en.cppreference.com/w/cpp/container/%s'))

def autolink(pattern, prefix=''):
    def role(name, rawtext, text, lineno, inliner, options={}, content=[]):
        url = pattern % (text,)
        node = nodes.reference(rawtext, prefix + text, refuri=url, **options)
        return [node], []
    return role
