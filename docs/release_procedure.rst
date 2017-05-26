.. Copyright (c) 2007-2017 Louisiana State University

   Distributed under the Boost Software License, Version 1.0. (See accompanying
   file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

-------------------------
Release Procedure for HPX
-------------------------

Below is a step-wise procedure for making an HPX release.

This is a living document and may not be totally current or accurate.
It is an attempt to capture current practice in making an HPX release.
Please update it as appropriate.

One way to use this procedure is to print a copy and check off
the lines as they are completed to avoid confusion.

#.  Notify developers that a release is imminent.

#.  Make a list of examples and benchmarks that should not go into the release.
    Build all examples and benchmarks that will go in the release and make sure
    they build and run as expected.

    *   Make sure all examples and benchmarks have example input files, and
        usage documentation, either in the form of comments or a readme.

#.  Send the list of examples and benchmarks that will be included in the
    release to hpx-users@stellar.cct.lsu.edu and stellar@cct.lsu.edu, and ask
    for feedback. Update the list as necessary.

#.  Write release notes for the blog to summarize the major changes listed in
    the log. The blog article should go in the "downloads" section. The url of
    the blog article should follow this format (where # are version numbers):
    stellar.cct.lsu.edu/downloads/hpx-v#-#-#-release-notes

    *   Save the blog article as a draft. Place the release notes into a new section
        in ``docs/whats_new.qbk``.

#.  Build the docs, and proof-read them. Update any documentation that may have
    changed, and correct any typos. Pay special attention to:

    *   ``$HPX_SOURCE/README.rst``
         *   Update grant information
    *   ``docs/whats_new.qbk``
    *   ``docs/people.qbk``
         *   Update collaborators
         *   Update grant information

#.  If there have been any commits to the release branch since the last release
    create a tag from the old release branch before deleting the old release
    branch in the next step.

#.  Delete the old release branch, and create a new one by branching a stable
    point from master.

    *   ``git push origin --delete [branch name]``
    *   ``git branch -D [branch name]``
    *   ``git branch [new branch name]``
    *   ``git push origin [new branch name]``
    *   ``git branch --set-upstream-to=origin/[branch name] [branch name]``

#.  Checkout the release branch, and remove the ``-trunk`` tag from
    ``hpx/config/version.hpp`` (replace it with ``-rc1`` for the release
    and later with an empty string for the actual release).

#.  Change logo for release documentation by removing '_draft' suffix
    in ``docs/cmakelist.txt`` on line 234. Update logo size accordingly on
    lines 330/331.

#.  Remove the examples and benchmarks that will not go into the release from
    the release branch.

#.  Remove features which have been deprecated for at least 2 releases. This
    involves removing build options which enable those features from the main
    CMakeLists.txt and also deleting all related code and tests from the main
    source tree.

    The general deprecation policy involves a three-step process we have to go
    through in order to introduce a breaking change

    a. First release cycle: add a build option which allows to explicitly disable
       any old (now deprecated) code.
    b. Second release cycle: turn this build option OFF by default.
    c. Third release cycle: completely remove the old code.

    The main CMakeLists.txt contains a comment indicating for which version
    the breaking change was introduced first.

#.  Tag a release candidate from the release branch.

    *   ``git tag -a [tag name] -m '[description]'``
    *   ``git push origin [tag name]``
    *   Create a pre-release on GitHub

#.  Switch Buildbot over to test the release branch

    *   https://github.com/STEllAR-GROUP/hermione-buildbot/blob/master/master/master.cfg
    *   Line 120

#.  Notify hpx-users@stellar.cct.lsu.edu and stellar@cct.lsu.edu of the
    availability of the release candidate. Ask users to test the candidate by
    checking out the release candidate tag.

#.  Allow at least a week for testing of the release candidate.

    *   Use ``git merge`` when possible, and fall back to ``git cherry-pick``
        when needed.

#.  Checkout the main branch, and bump the HPX version to the next release
    target. The following files contain version info:

    *   ``hpx/config/version.hpp``
    *   ``docs/hpx.qbk``
    *   ``CMakeLists.txt``
    *   Grep for old version number

#.  Create new logos for documentation. Update the logo used on line 234
    (add '_draft') and change the size accordingly in ``docs/cmakelist.txt``
    lines 330/331.

#.  Update ``$HPX_SOURCE/README.rst``

    *   Update version
    *   Update links to documentation

#.  Push changes to new branch numbered after the next release (not the current
    one).

#.  Tag the release.

#.  Roll a release candidate using ``tools/roll_release.sh`` (from root directory), and add the
    hashsums generated by the script to the "downloads" page of the website.

#.  Post the draft of the release notes.

#.  Upload the packages and generated documentation to the website. Use the following
    formats::

        http://stellar.cct.lsu.edu/files/hpx_#.#.#.zip
        http://stellar.cct.lsu.edu/files/hpx_#.#.#.tar.gz
        http://stellar.cct.lsu.edu/files/hpx_#.#.#.tar.bz2
        http://stellar.cct.lsu.edu/files/hpx_#.#.#.7z
        http://stellar.cct.lsu.edu/files/hpx_#.#.#/html
        http://stellar.cct.lsu.edu/files/hpx_#.#.#/html/code
        http://stellar.cct.lsu.edu/downloads/hpx-v#-#-#-release-notes

#.  Write a new blog post announcing the release.

#.  Create a release on github

    *   A DOI number using Zenodo is automatically assigned once the release is
        created as such on github.
    *   Verify on Zenodo (https://zenodo.org/) that release was uploaded.
        Logging into zenodo using the github credentials might be necessary to
        see the new release as it usually takes a while for it to propagate to
        the search engine used on zenodo.
    *   Fix zenodo reference number in main Readme.rst on the branch which holds
        the versioning changes.

#.  Merge release branch into master.

#.  Merge new branch containing next version numbers to master, resolve conflicts
    if necessary.

#.  Announce the release on hpx-users@stellar.cct.lsu.edu,
    stellar@cct.lsu.edu, allcct@cct.lsu.edu, faculty@csc.lsu.edu, faculty@ece.lsu.edu,
    xpress@crest.iu.edu, Sonia Sachs, our list of external collaborators,
    isocpp.org, HPC Wire, Inside HPC, and a CCT press release.

#.  Beer and pizza.

