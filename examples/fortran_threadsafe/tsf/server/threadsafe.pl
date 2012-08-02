#!/usr/bin/perl -w
# Copyright (c) 2007--2012 Matthew Anderson
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
@cpp_files = qw(point.cpp);
@fortran_files = qw(setup.F90);

$duplicates = 12;
if ( $#ARGV == 0 ) {
  $duplicates = $ARGV[0];
}
print "Number of duplications of fortran modules requested: $duplicates\n";

# The first thing we do is look through the fortran code to see if 
# there are any modules used;  if so, we rename both the module and the
# routine that uses it
$subroutine = "";
$sc = 0;
foreach $file (@fortran_files) {
  open(IN,"<$file") || die "Can't open $file \n";
  while (<IN>) {
    chomp;
    tr/A-Z/a-z/;
    # make sure we aren't dealing with a commented line 
    if (/!/) {
    } else {
      if (/subroutine/) {
        # get the name of the actual subroutine being examined
        $fin = $_;   
        $fin =~ s/subroutine //g;
        $fin =~ s/end//g;
        $fin =~ s/ //g;
        @result = split(/\(/,$fin);
        $subroutine = $result[0]; 
      }
      if (/c_ptr/) {
        # this one is already threadsafe, ignore
      } else {
        if (/use/) {
          # get the module name
          @r1 = split(/:/);
          @r2 = split(/\,/,$r1[0]);
          $module = $r2[0];   
          $module =~ s/use //g;
          $module =~ s/ //g;
          $subroutines[$sc] = $subroutine;
          $modules[$sc] = $module;
          $sc++;
        }
      }
    }
  }
  close(IN);
}

# get a unique list of the module names that are troublemakers
for ($i=0;$i<$sc;$i++) {
  $umm{$modules[$i]} = 1;
}

# get a unique list of the subroutine names that are troublemakers
for ($i=0;$i<$sc;$i++) {
  $usm{$subroutines[$i]} = 1;
}

# we now have two arrays that tell us which modules need to be made threadsafe
# and which subroutines use them.  Let's preprocess these files and duplicate the
# code in order to make it threadsafe.  This amounts to extracting each module
# and changing it's name to be unique.
$mcount = 0;
$capture = 0;
foreach $file (@fortran_files) {
  ($name,$suffix) = split(/\./,$file);
  $fileout = $name . "_pl." . $suffix;
  open(IN,"<$file") || die "Can't open $file \n";
  open(OUT,">$fileout") || die "Can't open $fileout \n";
  print OUT "! WARNING:  THIS FILE IS GENERATED FROM A PERL SCRIPT:  DO NOT MODIFY\n";
  print OUT "! SINCE YOUR CHANGES WILL NOT PERSIST BETWEEN BUILDS.\n";
  print OUT "!                                                    \n";
  while (<IN>) {
    chomp;
    tr/A-Z/a-z/;

    # make sure we aren't dealing with a commented line 
    if (/!/) {
    } else {
      #################################################
      #duplicate subroutines
      if (/subroutine/) {
        if (/end subroutine/) {
          # end of capture
          if ( $capture == 1 ) {
            $capture = 2;
            $src[$mcount] = $_;
            $mcount++;

            # duplicate the source code
            $newname = $subroutine_name . "_0";
            $src[0] =~ s/$subroutine_name/$newname/;
            $src[$mcount-1] =~ s/$subroutine_name/$newname/;
            for ($i=0;$i<$duplicates;$i++) {
              for ($j=0;$j<$mcount;$j++) {
                if ( ($j == 0 || $j == $mcount-1) && $i > 0 ) {
                  $k = $i-1;
                  $src[$j] =~ s/_$k/_$i/;
                  print OUT "$src[$j]\n"
                } else { 
                  # we need to change the 'use' statements also; they will show up here
                  if ($src[$j] =~ /use/) {
                    if ($src[$j] =~ /c_ptr/) {
                      print OUT "$src[$j]\n"
                    } else {
                      # modify the module name
                      foreach $key (keys %umm) {
                        $output = $src[$j];
                        $nn = $key . "_" . $i;
                        $output =~ s/$key/$nn/;
                      }
                      print OUT "$output\n";
                    }
                  } else {
                    print OUT "$src[$j]\n"
                  }
                }
              }
            }
          }

          $mcount = 0;
        } else {
          @sn = split(/\(/);
          $subroutine_name = $sn[0];
          # check the name of the module
          $subroutine_name =~ s/subroutine //g;
          $subroutine_name =~ s/\(//g;
          $subroutine_name =~ s/ //g;

          # see if this subroutine is in the troublemaker list
          $found = 0;
          for ($i=0;$i<$sc;$i++) {
            if ( $subroutine_name eq $subroutines[$i] ) {
              $found = 1;
              last;
            }
          }
          if ( $found == 1 ) {
            $capture = 1;
          }
        }
      }

      #################################################
      # duplicate modules
      if (/module/) {
        if (/end module/) {
          # end of capture
          if ( $capture == 1 ) {
            $capture = 2;
            $src[$mcount] = $_;
            $mcount++;

            # duplicate the source code
            for ($i=0;$i<$duplicates;$i++) {
              for ($j=0;$j<$mcount;$j++) {
                if ( $j == 0 || $j == $mcount-1 ) {
                  print OUT $src[$j] . "_" . $i . "\n";
                } else { 
                  print OUT "$src[$j]\n"
                }
              }
            }
          }

          $mcount = 0;
        } else {
          $module_name = $_;
          # check the name of the module
          $module_name =~ s/module //g;
          $module_name =~ s/ //g;

          # see if this module is in the troublemaker list
          $found = 0;
          for ($i=0;$i<$sc;$i++) {
            if ( $module_name eq $modules[$i] ) {
              $found = 1;
              last;
            }
          }
          if ( $found == 1 ) {
            $capture = 1;
          }
        }
      }

      if ($capture == 1) {
        $src[$mcount] = $_;
        $mcount++;
      }

    }

    if ( $capture == 0 ) {
      # print to screen what you see
      print OUT "$_\n";
    }

    # take care of last line
    if ( $capture == 2 ) {
      $capture = 0;
    }

  }
}

# Deal with the C++ code now

$capture = 0;
$count = 0;
$bnm = 0;
foreach $file (@cpp_files) {
  ($name,$suffix) = split(/\./,$file);
  $fileout = $name . "_pl." . $suffix;
  open(IN,"<$file") || die "Can't open $file \n";
  open(OUT,">$fileout") || die "Can't open $fileout \n";
  print OUT "// WARNING:  THIS FILE IS GENERATED FROM A PERL SCRIPT:  DO NOT MODIFY\n";
  print OUT "// SINCE YOUR CHANGES WILL NOT PERSIST BETWEEN BUILDS.\n";
  print OUT "//                                                    \n";
  while (<IN>) {
    chomp;

    if (/namespace/ && $bnm == 0 ) {
      $bnm = 1;
    }

    if ( $capture == 1 ) {
      $src[$count] = $_;
      $count++;
      if (/\;/) {
        $capture = 2;
        #duplicate
        foreach $key (keys %usm) {
          for ($i=0;$i<$duplicates;$i++) {
            if ( $bnm == 1 ) {
              print OUT "  if (item_ == $i) {\n";
            }

            for ($j=0;$j<$count;$j++) {
              $output = $src[$j];
              $nn = $key . "_" . $i;
              $output =~ s/$key/$nn/;
              print OUT "$output\n";
            }
            if ( $bnm == 1 ) {
              print OUT "  }\n";
            }
          }
        }
        $count = 0;
      }
    }
    if (/FNAME/) {
      # see if this fortran routine is one of the troublemakers
      foreach $key (keys %usm) {
        if (/$key/) {
          # it is; start to capture
          $capture = 1;
          $src[$count] = $_;
          $count++;
          if (/\;/) {
            # just one line -- no worries
            $capture = 2;
            #duplicate
            for ($i=0;$i<$duplicates;$i++) {
              for ($j=0;$j<$count;$j++) {
                $output = $src[$j];
                $nn = $key . "_" . $i;
                $output =~ s/$key/$nn/;
                print OUT "$output\n";
              }
            }
            $count = 0;
          }
        }
      }
      
    }
    if ( $capture == 0 ) {
      print OUT "$_\n";
    }
    if ( $capture == 2 ) {
      $capture = 0;
    }
#    # check to see if we are calling any fortran routines
#    if ( $active_grab == 1 ) {
#     $command[$count] = $_; 
#      $count++;
#      if (/\;/) {
#        # the entire command is in this line
#        $active_grab = 0;
# 
#        # print out the command
#       for ($i=0;$i<$count;$i++) {
#          print "$command[$i]\n";
#        }
#        $count = 0;
#      } else {
#       $active_grab = 1;
#      }
#    }
#    if (/FNAME/) {
#      # duplicate this code and modify function call
#      # get the entire function call -- it could extend for several lines
#      if (/\;/) {
#        # the entire command is in this line
#      } else {
#        $active_grab = 1;
#        $command[$count] = $_; 
#        $count++;
#      }
#    } 
#    print OUT "$_\n";
  }
  close(IN);
  close(OUT);
}

# Rename the original files so they aren't compiled
foreach $file (@cpp_files) {
  $nn = $file . ".orig";
  rename($file,$nn);
}
foreach $file (@fortran_files) {
  $nn = $file . ".orig";
  rename($file,$nn);
}
