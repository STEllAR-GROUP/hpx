load evs.mat
load A0.mat

function cmp_r = compare_floating (x, y, epsilon)
    if (((x + epsilon) >= y) && ((x - epsilon) <= y))
        cmp_r = 1;
    else
        cmp_r = 0;
    endif
endfunction

function are_r = compare_are (recorded, actual, tolerance)
    if (compare_floating(0.0, recorded, 1e-6))
        are_r = (tolerance >= (abs(recorded - actual) * 100.0));
    else
        are_r = (tolerance >= ((abs(recorded - actual) / recorded) * 100.0));
    endif
endfunction

verify_evs = sort(eig(A0))
evs = sort(evs)

for i = 1:rows(evs)
    compare_are(real(evs(i)), real(verify_evs(i)), 1.0) && \
    compare_are(imag(evs(i)), imag(verify_evs(i)), 1.0)
endfor

