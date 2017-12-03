#  Copyright (c) 2017 John Biddiscombe
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# ------------------------------------------------------
# pycicle
# Python Continuous Integration Command Line Engine
# Simple tool to poll PRs on github and spawn builds
# ------------------------------------------------------

from github import Github
import os, subprocess, time, re, string, random
from pathlib import Path

#----------------------------------------------------------------------------
# user's home dir and pycicle root
#----------------------------------------------------------------------------
home = str(Path.home())
pycicle_dir = os.environ.get('PYCICLE_ROOT', home + '/pycicle')
print('Pycicle using root path', pycicle_dir)

user_token = 'generate a token and paste it here, or set env var'
user_token = os.environ.get('GITHUB_TOKEN', user_token)

#----------------------------------------------------------------------------
# Create a Github instance:
#----------------------------------------------------------------------------
reponame  = 'HPX'
orgname   = 'STEllAR-GROUP'
poll_time = 60

git  = Github(orgname, user_token)
org  = git.get_organization(orgname)
repo = org.get_repo(reponame)

#----------------------------------------------------------------------------
# TODO : add support for multiple machines and configs
#----------------------------------------------------------------------------
current_path = os.path.dirname(os.path.realpath(__file__))
nickname = 'greina'
compiler = 'gcc'
boost = '1.65.1'

#----------------------------------------------------------------------------
# read one value from the CMake config for use elsewhere
#----------------------------------------------------------------------------
def get_setting_for_machine(machine, setting) :
    f = open(current_path + '/config/' + machine + '.cmake')
    for line in f:
        m = re.findall(setting + ' \"(.+?)\"', line)
        if m:
            print(setting, ' = ', m[0])
            return m[0]
    return ''

#----------------------------------------------------------------------------
# launch a script that will do one build
#----------------------------------------------------------------------------
def launch_pr_build(pr_number) :
  print('Launching build for PR', str(pr_number))

  remote_ssh  = get_setting_for_machine(nickname, 'PYCICLE_MACHINE')
  remote_path = get_setting_for_machine(nickname, 'PYCICLE_ROOT')
  remote_http = get_setting_for_machine(nickname, 'PYCICLE_HTTP')

  cmd = [current_path + '/launch_build.sh',
         remote_ssh, remote_path, nickname, str(pr_number), random_string(10),
         compiler, boost]

  print('Executing ', current_path + '/launch_build.sh')
  p = subprocess.Popen(cmd)
  print("pid = ", p.pid)
  return None

#----------------------------------------------------------------------------
# random string of N chars
#----------------------------------------------------------------------------
def random_string(N):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))

#----------------------------------------------------------------------------
# main polling routine
#----------------------------------------------------------------------------
while True:
    for pr in repo.get_pulls('open'):
        if not pr.mergeable:
            continue
        pr_str = str(pr.number)
        #print('git fetch origin pull/' + pr_str + '/head:pull_' + pr_str)
        directory = pycicle_dir + '/src/' + pr_str
        last_sha  = directory + '/last_build.txt'
        update = False
        if not os.path.exists(directory):
            os.makedirs(directory)
            if not os.path.is_file(last_sha):
                open(last_sha ,'w').write(pr.head.sha + '\n')
                update = True
            else:
                if pr.head.sha != open(last_sha).read():
                    update = True
        if update:
            launch_pr_build(pr.number)
        else:
#            print(pr.number, 'is up to date', pr.merge_commit_sha)
            ...

    # Sleep for a while before polling github again
    time.sleep(poll_time)
