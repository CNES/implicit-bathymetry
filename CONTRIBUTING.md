# How to contribute to implicit_bathy ?

Thank you for taking the time to contribute to implicit_bathy! This document will guide you
through the workflow and best practices you need to know to send your
contribution.

There are many ways to contribute to implicit_bathy:

* Reporting a bug
* Making a feature request
* Improving documentation
* Contributing code (Python, conda, etc.)

Our main workflow is on GitHub via the link :

[`https://.../implicit_bathy`](https://.../implicit_bathy)


## Reporting bugs

If you have found a bug, you can first [search the existing issues](https://implicit_bathy/-/issues)
to see if it has already been reported.

If it's a new bug, please [open a new issue](https://implicit_bathy/-/issues/new).
Try to be as precise as possible, about version, errors and the issue encountered.
Remember to add as much information as possible!

## Feature requests and discussions

Feature requests are welcome! Generally you are welcome to simply [open an issue](https://implicit_bathy/-/issues/new)
and discuss your idea there.

## Documentation improvements

The main implicit_bathy documentation is
the [online documentation](https://implicit_bathy/readme).  The source is
hosted in the main implicit_bathy repository in the `doc` directory. Therefore, to
contribute documentation use the same workflow as for code contributions (see
below).

See also the readme in `implicit_bathy/doc/` directory for help on building the Sphinx source.

## Code contribution

The implicit_bathy workflow is based on [Merge Requests](https://implicit_bathy/-/merge_requests).
Clone the repository, create a feature branch, commit your changes, push the
feature branch to a fork (or the main repository if you prefer),
then send a merge request.

Feature branches will be tested on the implicit_bathy tests (unit test and integration tests).
Tests must be launched before removing the 'draft' label to the MR. To launch the tests, ensure you have an internet connection then:

	cd your_implicit_bathy_clone/tests
    pytest -v -s --path_yaml tests/input/config_global.yaml --tmp_path tests/

### Commit message

On your feature branch, write a good [commit message](https://xkcd.com/1296/):
short and descriptive. If fixing an issue or bug, put the issue number in the
commit message so that Gitlab can [cross-link it](https://docs.gitlab.com/ce/user/project/issues/crosslinking_issues.html).
You can prefix your commit message with an indicating flag (DOC, BUG, TEST, etc.).

Standard prefixes for implicit lake commit messages:

    BUG: Fix for runtime crash or incorrect result
	BUG Fix: a bug mentioned in issue was solved by this commit
    DOC: Documentation change
    ENH: Enhancement of a function
	NEW FEATURE: New functionality
    PERF: Performance improvement
    STYLE: No logic impact (indentation, comments)
    TEST: Tests change
	WIP: Work In Progress not ready for merge

For example, here are some good commit messages:

    BUG FIX: #1701 Warn users if parameter string is unset
    DOC: Fix typo in home page
    BUG: variable not initialized if file format is different than sqlite

### Merge request

Your contribution is ready to be added to the main implicit_bathy repository? Send a Merge
Request against the `develop` branch on Framagit using the merge request
template. The merge request will then be discussed by the community and the core
implicit_bathy team : Santiago Pena Luque or Dawa Derksen (CNES)

* Merge requests can not be merged until all discussions have been resolved
* The merger is responsible for checking that the branch is up-to-date with develop
* Merge requests are merged only by the Merge Master (MM) or by any person who can push on `develop` and who has received the explicit agreement from the MM for the relevant MR.


### Contribution license agreement

implicit_bathy requires that contributors sign out a [Contributor License
Agreement](https://en.wikipedia.org/wiki/Contributor_License_Agreement). The
purpose of this CLA is to ensure that the project has the necessary ownership or
grants of rights over all contributions to allow them to distribute under the
chosen license (Affero General Public License 3.0).

To accept your contribution, we need you to complete, sign and email to cnes  fr* an [Individual Contributor Licensing
Agreement](https://implicit_bathy/.../assets/icla-en.odt) (ICLA) form and a
[Corporate Contributor Licensing
Agreement](https://implicit_bathy/.../assets/ccla-en.odt) (CCLA) form if you are
contributing on behalf of your company or another entity which retains copyright
for your contribution.


## Gitlab guidelines

In order to organize the issues in our gitlab instance, we use labels:


Regarding labels, we use the following set:

* ~Bug: for an unexpected behabior or crash during runtime
* ~CodeStyle: for major change looking at pep
* ~Documentation: for all lacks or errors in documentation
* ~Doing: the issue is currently in work
* ~FAQ: a recurrent question which should be added to the FAQ
* ~Install: an issue concerning installation or compilation
* ~Investigate: the issue need researchs or thinking to be solved, or for new package to be added
* ~Question: a question about a feature, a behavior, asking for clarification
* ~Refactoring: code which should be rewrite
* ~"To Do": almost all issue, the work should be done
* ~incident: issue during chain processing
* ~infrastructure: issue about system, internal lib used,...
* ~perfopt: issue to optimise performances
