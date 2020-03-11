..
    Copyright (c) 2007-2017 Louisiana State University

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

===========================
Release procedure for |hpx|
===========================

Below is a step by step procedure for making an |hpx| release. We aim to produce two
releases per year: one in March-April, and one in September-October.

This is a living document and may not be totally current or accurate. It is an
attempt to capture current practices in making an |hpx| release. Please update it
as appropriate.

One way to use this procedure is to print a copy and check off the lines as they
are completed to avoid confusion.

#. Notify developers that a release is imminent.

#. Make a list of examples and benchmarks that should not go into the release.
   Build all examples and benchmarks that will go in the release and make sure
   they build and run as expected.

    * Make sure all examples and benchmarks have example input files, and usage
      documentation, either in the form of comments or a readme.

#. Send the list of examples and benchmarks that will be included in the release
   to hpx-users@stellar.cct.lsu.edu and stellar@cct.lsu.edu, and ask for
   feedback. Update the list as necessary.

#. Write release notes in ``docs/sphinx/releases/whats_new_$VERSION.rst``. Keep
   adding merged PRs and closed issues to this until just before the release is
   made. Use ``tools/generate_pr_issue_list.sh`` to generate the lists. Add the
   new release notes to the table of contents in ``docs/sphinx/releases.rst``.

#. Build the docs, and proof-read them. Update any documentation that may have
   changed, and correct any typos. Pay special attention to:

   * ``$HPX_SOURCE/README.rst``

     * Update grant information

   * ``docs/sphinx/releases/whats_new_$VERSION.rst``
   * ``docs/sphinx/about_hpx/people.rst``

     *   Update collaborators
     *   Update grant information

#. This step does not apply to patch releases. For both APEX and hpxMP:

   * Change the release branch to be the most current release tag available in
     the APEX/hpxMP ``git_external`` section in the main ``CMakeLists.txt``.
     Please contact the maintainers of the respective packages to generate a new
     release to synchronize with the |hpx| release (`APEX
     <http://github.com/khuck/xpress-apex>`_, `hpxMP
     <https://github.com/STEllAR-GROUP/hpxMP>`_).

#. If there have been any commits to the release branch since the last release,
   create a tag from the old release branch before deleting the old release
   branch in the next step.

#. Unprotect the release branch in the github repository settings so that it can
   be deleted and recreated.

#. Delete the old release branch, and create a new one by branching a stable
   point from master. If you are creating a patch release, branch from the
   release tag for which you want to create a patch release.

   * ``git push origin --delete release``
   * ``git branch -D release``
   * ``git checkout [stable point in master]``
   * ``git branch release``
   * ``git push origin release``
   * ``git branch --set-upstream-to=origin/release release``

#. Protect the release branch again to disable deleting and force pushes.

#. Check out the release branch.

#. Make sure ``HPX_VERSION_MAJOR/MINOR/SUBMINOR`` in ``CMakeLists.txt`` contain
   the correct values. Change them if needed.

#. This step does not apply to patch releases. Remove features which have been
   deprecated for at least 2 releases. This involves removing build options
   which enable those features from the main CMakeLists.txt and also deleting
   all related code and tests from the main source tree.

   The general deprecation policy involves a three-step process we have to go
   through in order to introduce a breaking change:

   a. First release cycle: add a build option that allows for explicitly disabling
      any old (now deprecated) code.
   b. Second release cycle: turn this build option OFF by default.
   c. Third release cycle: completely remove the old code.

   The main CMakeLists.txt contains a comment indicating for which version
   the breaking change was introduced first.

#. Update the minimum required versions if necessary (compilers, dependencies,
   etc.).

#. Switch Buildbot over to test the release branch.

   * ``https://github.com/STEllAR-GROUP/hermione-buildbot/blob/rostam/master/master.cfg``
   * ``branch`` field in ``c['change_source'] =  GitPoller``

#. Repeat the following steps until satisfied with the release.

   #. Change ``HPX_VERSION_TAG`` in ``CMakeLists.txt`` to ``-rcN``, where ``N``
      is the current iteration of this step. Start with ``-rc1``.

   #. Tag and create a pre-release on GitHub using the script
      ``tools/roll_release.sh``. The script requires that you have the |stellar|
      Group signing key.

   #. This step is not necessary for patch releases. Notify
      ``hpx-users@stellar.cct.lsu.edu`` and ``stellar@cct.lsu.edu`` of the
      availability of the release candidate. Ask users to test the candidate by
      checking out the release candidate tag.

   #. Allow at least a week for testing of the release candidate.

      * Use ``git merge`` when possible, and fall back to ``git cherry-pick``
        when needed. For patch releases ``git cherry-pick`` is most likely your
        only choice if there have been significant unrelated changes on master
        since the previous release.
      * Go back to the first step when enough patches have been added.
      * If there are no more patches, continue to make the final release.

#. Update any occurrences of the latest stable release to refer to the version
   about to be released. For example, ``quickstart.rst`` contains instructions
   to check out the latest stable tag. Make sure that refers to the new version.

#. Add a new entry to the RPM changelog (``cmake/packaging/rpm/Changelog.txt``)
   with the new version number and a link to the corresponding changelog.

#. Change ``HPX_VERSION_TAG`` in ``CMakeLists.txt`` to an empty string.

#. Add the release date to the caption of the current "What's New" section in
   the docs, and change the value of ``HPX_VERSION_DATE`` in
   ``CMakeLists.txt``.

#. Tag and create a release on GitHub using the script
   ``tools/roll_release.sh``. The script requires that you have the |stellar|
   Group signing key.

#. Update the websites (`stellar-group.org <https://stellar-group.org>`_ and
   `stellar.cct.lsu.edu <https://stellar.cct.lsu.edu>`_) with the following:

   * Download links on the downloads pages. Link to the release on GitHub.
   * Documentation links on the docs page (link to generated documentation on
     GitHub Pages). Follow the style of previous releases.
   * A new blog post announcing the release, which links to downloads and the
     "What's New" section in the documentation (see previous releases for
     examples).

#. Merge release branch into master.

#. Post-release cleanup. Create a new pull request against master with the
   following changes:

   #. Modify the release procedure if necessary.

   #. Change ``HPX_VERSION_TAG`` in ``CMakeLists.txt`` back to ``-trunk``.

#. Switch Buildbot back to test the main branch

   * ``https://github.com/STEllAR-GROUP/hermione-buildbot/blob/rostam/master/master.cfg``
   * ``branch`` field in ``c['change_source'] =  GitPoller``

#. Update Vcpkg (``https://github.com/Microsoft/vcpkg``) to pull from latest release.

   * Update version number in CONTROL
   * Update tag and SHA512 to that of the new release

#. Announce the release on hpx-users@stellar.cct.lsu.edu, stellar@cct.lsu.edu,
   allcct@cct.lsu.edu, faculty@csc.lsu.edu, faculty@ece.lsu.edu,
   xpress@crest.iu.edu, the |hpx| Slack channel, the IRC channel, Sonia Sachs,
   our list of external collaborators, isocpp.org, reddit.com, HPC Wire, Inside
   HPC, Heise Online, and a CCT press release.

#. Beer and pizza.

