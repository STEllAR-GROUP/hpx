.. Copyright (c) 2007-2017 Louisiana State University

   Distributed under the Boost Software License, Version 1.0. (See accompanying
   file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

===========================
Release procedure for |hpx|
===========================

Below is a step-wise procedure for making an HPX release. We aim to produce two
releases per year: one in March-April, and one in September-October.

This is a living document and may not be totally current or accurate. It is an
attempt to capture current practice in making an HPX release. Please update it
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
   made. Add the new release notes to the table of contents in
   ``docs/sphinx/releases.rst``.

#. Build the docs, and proof-read them. Update any documentation that may have
   changed, and correct any typos. Pay special attention to:

   * ``$HPX_SOURCE/README.rst``

     * Update grant information

   * ``docs/sphinx/releases/whats_new_$VERSION.rst``
   * ``docs/sphinx/about_hpx/people.rst``

     *   Update collaborators
     *   Update grant information

#. Change the APEX release branch to be the most current release tag available
   in the ``git_external`` APEX section of the main ``CMakeLists.txt``. Please
   contact the maintainers at the `APEX repository
   <http://github.com/khuck/xpress-apex>`_ to generate a new release to
   synchronize with the HPX release.

#. Change the hpxMP release branch to be the most current release tag available
   in the ``git_external`` hpxMP section of the main ``CMakeLists.txt``. Please
   contact the maintainers at the `hpxMP repository
   <https://github.com/STEllAR-GROUP/hpxMP>`_ to generate a new release to
   synchronize with the HPX release.

#. If there have been any commits to the release branch since the last release
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

#. Checkout the release branch, and replace the ``-trunk`` tag in
   ``CMakeLists.txt`` with ``-rc1``.

#. Remove the examples and benchmarks that will not go into the release from the
   release branch.

#. Remove features which have been deprecated for at least 2 releases. This
   involves removing build options which enable those features from the main
   CMakeLists.txt and also deleting all related code and tests from the main
   source tree. This step does not apply to patch releases.

   The general deprecation policy involves a three-step process we have to go
   through in order to introduce a breaking change

   a. First release cycle: add a build option which allows to explicitly disable
      any old (now deprecated) code.
   b. Second release cycle: turn this build option OFF by default.
   c. Third release cycle: completely remove the old code.

   The main CMakeLists.txt contains a comment indicating for which version
   the breaking change was introduced first.

#. Tag a release candidate from the release branch, where tag name is the
   version to be released with a "-rcN" suffix and description is
   "HPX V$VERSION: The C++ Standards Library for Parallelism and Concurrency".

   * ``git tag -a [tag name] -m '[description]'``
   * ``git push origin [tag name]``
   * Create a pre-release on GitHub

#. Switch Buildbot over to test the release branch

   * ``https://github.com/STEllAR-GROUP/hermione-buildbot/blob/rostam/master/master.cfg``
   * Line 120

#. Notify ``hpx-users@stellar.cct.lsu.edu`` and ``stellar@cct.lsu.edu`` of the
   availability of the release candidate. Ask users to test the candidate by
   checking out the release candidate tag.

#. Allow at least a week for testing of the release candidate.

   * Use ``git merge`` when possible, and fall back to ``git cherry-pick`` when
     needed.
   * Repeat by tagging a new release candidate as many times as needed.

#. Checkout the release branch, and replace the ``-rcN`` tag in
   ``CMakeLists.txt`` with an empty string.

#. Update any occurrences of the latest stable release to refer to the version
   about to be released. For example, ``quickstart.rst`` contains instructions
   to check out the latest stable tag. Make sure that refers to the new version.

#. Add a new entry to the RPM changelog (``cmake/packaging/rpm/Changelog.txt``)
   with the new version number and a link to the corresponding changelog.

#. Add the release date to the caption of the current "What's New" section in
   the docs, and change the value of ``HPX_VERSION_DATE`` in
   ``CMakeLists.txt``.

#. Tag the release from the release branch, where tag name is the version to be
   released and description is "HPX V$VERSION: The C++ Standards Library for
   Parallelism and Concurrency". Sign the release tag with the
   ``contact@stellar-group.org`` key by adding the ``-s`` flag to ``git tag``.
   Make sure you change git to sign with the ``contact@stellar-group.org`` key,
   rather than your own key if you use one. You also need to change the name and
   email used for commits. Change them to ``STE||AR Group`` and
   ``contact@stellar-group.org``, respectively. Finally, the
   ``contact@stellar-group.org`` email address needs to be added to your GitHub
   account for the tag to show up as verified.

   * ``git tag -s -a [tag name] -m '[description]'``
   * ``git push origin [tag name]``

#. Create a release on github

   * Refer to 'What's New' section in the documentation you uploaded in the
     notes for the Github release (see previous releases for a hint).
   * A DOI number using Zenodo is automatically assigned once the release is
     created as such on github.
   * Verify on Zenodo (https://zenodo.org/) that release was uploaded. Logging
     into zenodo using the github credentials might be necessary to see the new
     release as it usually takes a while for it to propagate to the search
     engine used on zenodo.

#. Roll a release candidate using ``tools/roll_release.sh`` (from root
   directory), and add the hashsums generated by the script to the "downloads"
   page of the website.

#. Upload the packages the website. Use the following formats:

   .. code-block:: text

      http://stellar.cct.lsu.edu/files/hpx_#.#.#.zip
      http://stellar.cct.lsu.edu/files/hpx_#.#.#.tar.gz
      http://stellar.cct.lsu.edu/files/hpx_#.#.#.tar.bz2
      http://stellar.cct.lsu.edu/files/hpx_#.#.#.7z

#. Update the websites (`stellar-group.org <https://stellar-group.org>`_ and
   `stellar.cct.lsu.edu <https://stellar.cct.lsu.edu>`_) with the following:

   * Download links on the download page
   * Documentation links on the docs page (link to generated documentation on
     GitHub Pages)
   * A new blog post announcing the release, which links to downloads and the
     "What's New" section in the documentation (see previous releases for examples)

#. Merge release branch into master.

#. Create a new branch from master, and check that branch out (name it for
   example by the next version number). Bump the HPX version to the next
   release target. The following files contain version info:

   * ``CMakeLists.txt``
   * Grep for old version number

#. Create a new "What's New" section for the docs of the next anticipated
   release. Set the date to "unreleased".

#. Merge new branch containing next version numbers to master, resolve conflicts
   if necessary.

#. Switch Buildbot back to test the main branch

   * ``https://github.com/STEllAR-GROUP/hermione-buildbot/blob/rostam/master/master.cfg``
   * Line 120

#. Update Vcpkg (``https://github.com/Microsoft/vcpkg``) to pull from latest release.

   * Update version number in CONTROL
   * Update tag and SHA512 to that of the new release

#. Announce the release on hpx-users@stellar.cct.lsu.edu, stellar@cct.lsu.edu,
   allcct@cct.lsu.edu, faculty@csc.lsu.edu, faculty@ece.lsu.edu,
   xpress@crest.iu.edu, the |hpx| Slack channel, the IRC channel, Sonia Sachs,
   our list of external collaborators, isocpp.org, reddit.com, HPC Wire, Inside
   HPC, Heise Online, and a CCT press release.

#. Beer and pizza.

