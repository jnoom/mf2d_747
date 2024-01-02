function out = vaf(y,yhat)
out = max(0, 100 * (1 - var(y-yhat) / var(y)));
end

