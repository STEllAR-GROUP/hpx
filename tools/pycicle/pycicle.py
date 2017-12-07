#  Copyright (c) 2017 John Biddiscombe
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# ------------------------------------------------------
# pycicle
# Python Continuous Integration Command Line Engine
# Simple tool to poll PRs on github and spawn builds
# ------------------------------------------------------
from __future__ import absolute_import, division, print_function, unicode_literals
import github, os, subprocess, time, re, string, random, socket, datetime

#----------------------------------------------------------------------------
# user's home dir and pycicle root
#----------------------------------------------------------------------------
home = str(os.path.expanduser('~'))
pycicle_dir = os.environ.get('PYCICLE_ROOT', home + '/pycicle')
print('pycicle using root path', pycicle_dir)

#----------------------------------------------------------------------------
# github token used to authenticate access
#----------------------------------------------------------------------------
user_token = 'generate a token and paste it here, or set env var'
user_token = os.environ.get('PYCICLE_GITHUB_TOKEN', user_token)

#----------------------------------------------------------------------------
# Machines : get a list of machines to use for testing from PYCICLE_MACHINES
# please use a comma separated list of machine nicknames such as
# greina;daint;jb-laptop
# where the names corresond to the name.cmake files in the config dir
#
# TODO : add support for multiple machines and configs
#----------------------------------------------------------------------------
machines     = os.environ.get('PYCICLE_MACHINES', 'greina')
machine_list = machines.split(',')
machine      = machine_list[0]
print('pycicle using machines', machine_list)
print('current implementation supports 1 machine :', machine)

#----------------------------------------------------------------------------
# Debuging - set PYCICLE_DEBUG env var to disable triggering builds
#----------------------------------------------------------------------------
debug_mode = os.environ.get('PYCICLE_DEBUG', '') != ''

#----------------------------------------------------------------------------
# Create a Github instance:
#----------------------------------------------------------------------------
reponame    = 'HPX'
orgname     = 'STEllAR-GROUP'
poll_time   = 60
scrape_time = 10*60

try:
    git  = github.Github(orgname, user_token)
    org  = git.get_organization(orgname)
    repo = org.get_repo(reponame)
except:
    print('Failed to connect to github. Network down?')

#----------------------------------------------------------------------------
# Scrape-list : machine/build that we must check for status files
# This will need to support lots of build/machine combinations eventually
#----------------------------------------------------------------------------
scrape_list = {}

#----------------------------------------------------------------------------
# read one value from the CMake config for use elsewhere
#----------------------------------------------------------------------------
def get_setting_for_machine(machine, setting) :
    current_path = os.path.dirname(os.path.realpath(__file__))
    f = open(current_path + '/config/' + machine + '.cmake')
    for line in f:
        m = re.findall(setting + ' \"(.+?)\"', line)
        if m:
            return m[0]
    return ''

#----------------------------------------------------------------------------
# launch a command that will start one build
#----------------------------------------------------------------------------
def launch_build(nickname, branch_id, branch_name) :
    remote_ssh  = get_setting_for_machine(nickname, 'PYCICLE_MACHINE')
    remote_path = get_setting_for_machine(nickname, 'PYCICLE_ROOT')
    remote_http = get_setting_for_machine(nickname, 'PYCICLE_HTTP')

    # we are not yet using these as 'options'
    compiler = 'xxx'
    boost = 'x.xx.x'

    cmd = ['ssh', remote_ssh, 'ctest -S ',
           remote_path          +'/repo/tools/pycicle/dashboard_slurm.cmake',
           '-DPYCICLE_ROOT='    + remote_path,
           '-DPYCICLE_HOST='    + nickname,
           '-DPYCICLE_PR='      + branch_id if branch_id != 'master' else '',
           '-DPYCICLE_BRANCH='  + branch_name,
           '-DPYCICLE_RANDOM='  + random_string(10),
           '-DPYCICLE_COMPILER='+ compiler,
           '-DPYCICLE_BOOST='   + boost,
           '-DPYCICLE_MASTER='  + 'master',
           # These are to quiet warnings from ctest about unset vars
           '-DCTEST_SOURCE_DIRECTORY=.',
           '-DCTEST_BINARY_DIRECTORY=.',
           '-DCTEST_COMMAND=":"']

    if debug_mode:
        print('\n' + '-' * 20, 'Debug\n', cmd)
        print('-' * 20 + '\n')
    else:
        print('\n' + '-' * 20, 'Executing\n', cmd)
        p = subprocess.Popen(cmd)
        print('-' * 20 + '\n')

    return None

#----------------------------------------------------------------------------
# collect test results so that we can update github PR status
#----------------------------------------------------------------------------
def scrape_testing_results(nickname, branch_id, branch_name, head_commit) :
    remote_ssh  = get_setting_for_machine(nickname, 'PYCICLE_MACHINE')
    remote_path = get_setting_for_machine(nickname, 'PYCICLE_ROOT')

    cmd = ['ssh', remote_ssh, 'cat ',
           remote_path + '/build/' + branch_id + '/pycicle-TAG.txt']

    Config_Errors = 0
    Build_Errors  = 0
    Test_Errors   = 0
    Errors        = []

    try:
        result = subprocess.check_output(cmd).split()
        print('Scrape result is', result)
        for s in result: Errors.append(s.decode('utf-8'))
        print('Errors are', Errors)

        Config_Errors = int(Errors[0])
        Build_Errors  = int(Errors[1])
        Test_Errors   = int(Errors[2])
        DateStamp     = Errors[3]
        DateURL       = DateStamp[0:4]+'-'+DateStamp[4:6]+'-'+DateStamp[6:8]
        print('Extracted date as ', DateURL)

        # if this file has been scraped before, then don't do it again
        if len(Errors)>4 and Errors[4]=="PYCICLE_GITHUB_STATUS_SET":
            print('Scrape not needed, status already set for', branch_id)
            return True

        URL = ('http://cdash.cscs.ch/index.php?project=HPX' +
               '&date=' + DateURL +
               '&filtercount=1' +
               '&field1=buildname/string&compare1=63&value1=' +
               branch_id + '-' + branch_name)

        if debug_mode:
            print ('Debug github PR status', URL)
        else:
            print ('Updating github PR status', URL)
            head_commit.create_status(
                'success' if Config_Errors==0 else 'failure',
                target_url=URL,
                description='errors ' + Errors[0],
                context='pycicle-Config')
            head_commit.create_status(
                'success' if Build_Errors==0 else 'failure',
                target_url=URL,
                description='errors ' + Errors[1],
                context='pycicle-Build')
            head_commit.create_status(
                'success' if Test_Errors==0 else 'failure',
                target_url=URL,
                description='errors ' + Errors[2],
                context='pycicle-Test')

        # update the pycicle scrape file if we have set status corectly
        try:
            cmd = ['ssh', remote_ssh, 'echo PYCICLE_GITHUB_STATUS_SET >>' +
                remote_path + '/build/' + branch_id + '/pycicle-TAG.txt']
            result = subprocess.check_output(cmd).split()
            print ('Scrape file updated', cmd)
        except:
            print ('Scrape file update failed', cmd)

        print ('Done scraping and setting github PR status')
        return True

    except:
        print('Scrape failed for PR', branch_id)

    return False

#----------------------------------------------------------------------------
# random string of N chars
#----------------------------------------------------------------------------
def random_string(N):
    return ''.join(random.choice(string.ascii_uppercase + string.digits)
        for _ in range(N))

#----------------------------------------------------------------------------
# Check if a PR Needs and Update
#----------------------------------------------------------------------------
def needs_update(branch_id, branch_name, branch_sha, master_sha):
    directory     = pycicle_dir + '/src/' + branch_id
    status_file   = directory + '/last_pr_sha.txt'
    update        = False
    #
    if not os.path.exists(directory):
        os.makedirs(directory)
        update = True
    else:
        try:
            f = open(status_file,'r')
            lines = f.readlines()
            if lines[0].strip() != branch_sha:
                print(branch_id, branch_name, 'changed : trigger update')
                update = True
            elif (lines[1].strip() != master_sha):
                print('master changed : trigger update')
                update = True
            f.close()
        except:
            print(branch_id, branch_name, 'status error : trigger update')
            update = True
    #
    if update:
        f = open(status_file,'w')
        f.write(branch_sha + '\n')
        f.write(master_sha + '\n')
        f.close()
    #
    return update

#----------------------------------------------------------------------------
# main polling routine
#----------------------------------------------------------------------------
if debug_mode:
    print('pycicle is in debug mode, no build trigger commands will be sent')
#
first_iteration = True
github_t1       = datetime.datetime.now()
scrape_t1       = github_t1 + datetime.timedelta(hours=-1)
#
while True:
    #
    try:
        github_t2     = datetime.datetime.now()
        github_tdiff  = github_t2 - github_t1
        github_t1     = github_t2
        print('Checking github:', 'Time since last check', github_tdiff.seconds, '(s)')
        master_branch = repo.get_branch(repo.default_branch)
        #
        for pr in repo.get_pulls('open'):
            if not pr.mergeable:
                continue
            #
            branch_id   = str(pr.number)
            branch_name = pr.head.label.rsplit(':',1)[1]
            branch_sha  = pr.head.sha
            master_sha  = master_branch.commit.sha
            #
            update = needs_update(branch_id, branch_name, branch_sha, master_sha)
            #
            if update:
                launch_build(machine, branch_id, branch_name)
                # get last commit on PR for setting status
                scrape_list[branch_id] = [machine, branch_id, branch_name, pr.get_commits().reversed[0]]

            elif (first_iteration):
                scrape_list[branch_id] = [machine, branch_id, branch_name, pr.get_commits().reversed[0]]

        # also build the master branch if it changes
        if needs_update('master', 'master', master_sha, master_sha):
            launch_build(machine, 'master', 'master')
            scrape_list['master'] = [machine, 'master', 'master', master_branch.commit]

    except (github.GithubException, socket.timeout) as ex:
        # github might be down, or there may be a network issue,
        # just go to the sleep statement and try again in a minute
        print('Github/Socket exception :', ex)
        first_iteration = True

    scrape_t2    = datetime.datetime.now()
    scrape_tdiff = scrape_t2 - scrape_t1
    if (scrape_tdiff.seconds > scrape_time):
        scrape_t1 = scrape_t2
        print('Scraping results:', 'Time since last check', scrape_tdiff.seconds, '(s)')
        # force a scrape list copy, remove keys from actual scrape_list during iteration
        for build in list(scrape_list):
            values = scrape_list[build]
            print('\n' + '-' * 20, 'Scraping', values[0], values[1], values[2])
            # if the scrape suceeds, remove the build from the scrape list
            if scrape_testing_results(values[0], values[1], values[2], values[3]):
                del scrape_list[build]
            print('-' * 20)

    # Sleep for a while before polling github again
    time.sleep(poll_time)
