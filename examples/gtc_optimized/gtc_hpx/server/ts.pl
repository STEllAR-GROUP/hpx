#!/usr/bin/perl -w
# Copyright (c) 2007--2012 Matthew Anderson
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# manual list of all modules in the code
@modules1 = qw(precision global_parameters particle_array particle_tracking field_array diagnosis_array particle_decomp rngdef rngf77 rng_gtc rng1);

@modules2 = qw(global_parameters particle_array particle_tracking field_array diagnosis_array particle_decomp rngdef rngf77 rng_gtc rng1);

@subroutines = qw(chargei load rng_number_d0 rng_number_d1 rng_number_d2 rng_number_d3 rng_number_s0 rng_number_s1 rng_number_s2 rng_number_s3 rng_gauss_d0 rng_gauss_d1 rng_gauss_d2 rng_gauss_d3 rng_gauss_s0 rng_gauss_s1 rng_gauss_s2 rng_gauss_s3 rng_set_seed_time rng_set_seed_int rng_set_seed_string rng_print_seed rng_init rng_step_seed random1 srandom1 random_array srandom_array rand_batch random_init decimal_to_seed string_to_seed set_random_seed seed_to_decimal next_seed3 next_seed rand_axc restart_write restart_read rand_num_gen_init set_random_zion get_random_number setup read_input_params broadcast_input_params set_particle_decomp tag_particles locate_tracked_particles write_tracked_particles hdf5out_tracked_particles smooth);
         

if ( $#ARGV < 0 ) {
  print "Usage: ts.pl <filename> <module set 1 or 2; module set 2 is only used for random number generator > [number duplicates]\n";
  exit;
} 

$file = $ARGV[0];
$moduleset = $ARGV[1];

$duplicates = 10;
#if ( $#ARGV = 1 ) {
#  $duplicates = $ARGV[1];
#}

print "Number of duplications of fortran modules requested: $duplicates\n";

# go through each fortran file and duplicate each module 
$mcount = 0;
($name,$suffix) = split(/\./,$file);
$fileout = $name . "_pl." . $suffix;
open(IN,"<$file") || die "Can't open $file \n";
open(OUT,">$fileout") || die "Can't open $fileout \n";
while (<IN>) {
  chomp;
  tr/A-Z/a-z/;
  
  # change everything to real*8 
  $_ =~ s/real\(wp\)/real\(8\)/g;

  $src[$mcount] = $_;
  $mcount++;
}
close(IN);

if ( $moduleset == 1 ) {
  @full = (@subroutines,@modules1);
} else {
  @full = (@subroutines,@modules2);
}
for ($j=0;$j<$duplicates;$j++) {
  for ($i=0;$i<$mcount;$i++) {
    $output = $src[$i];
    foreach $sub (@full) {
      $nn = $sub . "_" . $j;
      $output =~ s/$sub/$nn/;
    }
    print OUT "$output\n";
  }
}
close(OUT);
