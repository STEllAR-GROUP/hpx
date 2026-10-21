#!/usr/bin/env python3
# Copyright (c) 2026 Anshuman Agrawal
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

"""Check report publication decisions without Slurm or network access."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "lsu-perftests/entry.sh"


class PerformanceResults(unittest.TestCase):
    def run_entry(self, *, slurm=0, status=None, report=False, changed=False,
                  stale=False, pull_request=True, cdash_build_id=None,
                  upload=0, comment_body=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lane = root / ".jenkins/lsu-perftests"
            common = root / ".jenkins/common"
            lane.mkdir(parents=True)
            common.mkdir()
            shutil.copyfile(SCRIPT, lane / "entry.sh")
            (lane / "slurm-constraint-perftests.sh").write_text(
                "configuration_slurm_partition=test\n"
                "configuration_slurm_nodes=test\n")
            (common / "slurm.sh").write_text('''
hpx_slurm_cancel_previous() { return 0; }
hpx_slurm_run() {
    echo "${HPX_PERFTESTS_REPORT_ALL:-}" > report-all.txt
    if [[ -n "${BUILD_STATUS}" ]]; then
        echo "${BUILD_STATUS}" > jenkins-hpx-perftests-ctest-status.txt
    fi
    if [[ "${CREATE_REPORT}" == 1 ]]; then
        mkdir -p perftests-reports/reference-comparison
        echo current > perftests-reports/reference-comparison/index.html
        if [[ "${CREATE_CHANGED}" == 1 ]]; then
            echo "test executor +" > \\
                perftests-reports/reference-comparison/changed.txt
        fi
    fi
    return "${SLURM_STATUS}"
}
''')
            comment = lane / "comment_github.sh"
            comment.write_text(
                "#!/bin/sh\necho \"${PERFTEST_IMAGES_URL}\" > comment-called\n")
            comment.chmod(0o755)
            upload_stub = lane / "upload_images.sh"
            upload_stub.write_text(
                "#!/bin/sh\nprintf '%s\\n' \"$@\" > upload-args.txt\n"
                "[ \"${UPLOAD_STATUS}\" = 0 ] || exit 1\n"
                "echo https://images.example/abc/pr/123/7\n")
            upload_stub.chmod(0o755)
            github_status = common / "set_github_status.sh"
            github_status.write_text(
                "#!/bin/sh\nshift\nprintf '%s\\n' \"$@\" > status-args.txt\n")
            github_status.chmod(0o755)
            if cdash_build_id is not None:
                (root / "jenkins-hpx-perftests-cdash-build-id.txt").write_text(
                    cdash_build_id)
            sleep = root / "sleep"
            sleep.write_text("#!/bin/sh\nexit 0\n")
            sleep.chmod(0o755)
            if stale:
                (root / "jenkins-hpx-perftests-ctest-status.txt").write_text("1")
                old_report = root / "perftests-reports/reference-comparison"
                old_report.mkdir(parents=True)
                (old_report / "index.html").write_text("previous run")
            env = dict(os.environ, PATH=str(root) + ":" + os.environ["PATH"],
                       configuration_name="perftests", GIT_BRANCH="origin/master",
                       BUILD_STATUS="" if status is None else str(status),
                       SLURM_STATUS=str(slurm), CREATE_REPORT=str(int(report)),
                       CREATE_CHANGED=str(int(changed)),
                       UPLOAD_STATUS=str(upload), BUILD_NUMBER="7",
                       ghprbPullId="123" if pull_request else "",
                       GITHUB_TOKEN="dummy-token", GIT_COMMIT="branch-sha",
                       ghprbActualCommit="pr-sha",
                       ghprbPullLink="https://github.com/TheHPXProject/hpx/pull/123")
            env.pop("HPX_PERFTESTS_REPORT_ALL", None)
            if comment_body is not None:
                env["ghprbCommentBody"] = comment_body
            result = subprocess.run(
                [os.environ.get("HPX_TEST_BASH", "bash"), str(lane / "entry.sh")],
                cwd=root, env=env, capture_output=True, text=True, timeout=10)
            status_args = root / "status-args.txt"
            self.status_args = (status_args.read_text().splitlines()
                                if status_args.exists() else None)
            upload_args = root / "upload-args.txt"
            self.upload_args = (upload_args.read_text().splitlines()
                                if upload_args.exists() else None)
            report_all = root / "report-all.txt"
            self.report_all = (report_all.read_text().strip()
                               if report_all.exists() else None)
            comment_called = root / "comment-called"
            self.images_url = (comment_called.read_text().strip()
                               if comment_called.exists() else None)
            return result.returncode, comment_called.exists()

    def test_timeout_without_results_does_not_comment(self):
        self.assertEqual(self.run_entry(slurm=124), (124, False))

    def test_timeout_does_not_publish_stale_results(self):
        self.assertEqual(self.run_entry(slurm=124, stale=True), (124, False))

    def test_success_publishes_its_report(self):
        self.assertEqual(self.run_entry(status=0, report=True, changed=True),
                         (0, True))
        self.assertEqual(self.images_url, "https://images.example/abc/pr/123/7")

    def test_unchanged_report_does_not_comment(self):
        self.assertEqual(self.run_entry(status=0, report=True), (0, False))
        self.assertEqual(self.status_args[2], "success")

    def test_failure_without_changes_does_not_comment(self):
        self.assertEqual(self.run_entry(slurm=1, status=1, report=True),
                         (1, False))
        self.assertEqual(self.status_args[2], "failure")

    def test_failed_comparison_publishes_its_report(self):
        self.assertEqual(self.run_entry(slurm=1, status=1, report=True,
                                        changed=True), (1, True))
        self.assertEqual(self.upload_args, [
            "perftests-reports/reference-comparison", "123", "7"])
        self.assertEqual(self.images_url, "https://images.example/abc/pr/123/7")

    def test_failed_upload_publishes_the_report_without_plots(self):
        self.assertEqual(self.run_entry(slurm=1, status=1, report=True,
                                        changed=True, upload=1), (1, True))
        self.assertEqual(self.images_url, "")

    def test_build_failure_without_report_does_not_comment(self):
        self.assertEqual(self.run_entry(slurm=1, status=1), (1, False))

    def test_branch_failure_does_not_comment_on_a_pull_request(self):
        self.assertEqual(self.run_entry(slurm=1, status=1, report=True,
                                        changed=True, pull_request=False),
                         (1, False))

    def test_full_report_is_requested_by_a_comment(self):
        for comment in ("retest lsu perf full", "Retest LSU Perf Full\n",
                        "retest full"):
            with self.subTest(comment=comment):
                self.run_entry(status=0, comment_body=comment)
                self.assertEqual(self.report_all, "1")

    def test_other_comments_dont_request_a_full_report(self):
        for comment in (None, "", "retest lsu perf", "retest"):
            with self.subTest(comment=comment):
                self.run_entry(status=0, comment_body=comment)
                self.assertEqual(self.report_all, "")

    def test_success_reports_status_with_cdash_build(self):
        self.assertEqual(self.run_entry(status=0, cdash_build_id="42"),
                         (0, False))
        self.assertEqual(self.status_args, [
            "TheHPXProject/hpx", "pr-sha", "success", "perftests", "42",
            "jenkins/lsu-perftests"])

    def test_timeout_reports_failure_without_cdash_build(self):
        self.assertEqual(self.run_entry(slurm=124, pull_request=False),
                         (124, False))
        self.assertEqual(self.status_args, [
            "TheHPXProject/hpx", "branch-sha", "failure", "perftests", "",
            "jenkins/lsu-perftests"])

    def test_missing_status_is_a_failure(self):
        self.assertEqual(self.run_entry(), (1, False))


if __name__ == "__main__":
    unittest.main()
