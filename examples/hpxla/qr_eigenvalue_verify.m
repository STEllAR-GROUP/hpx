load evs.mat
load A0.mat

function retval = compare_floating (x, y, eps)
    if (((x + eps) >= y) && ((x - eps) <= y))
        retval = 1;
    else
        retval = 0;
    endif
endfunction

verify_evs = sort(eig(A0))
evs = sort(evs)

for i = 1:rows(evs)
    compare_floating(real(evs(i)), real(verify_evs(i)), 1e-2) && \
    compare_floating(imag(evs(i)), imag(verify_evs(i)), 1e-2)
endfor

