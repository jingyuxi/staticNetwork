function inverse_logit_stability(x_vec)
 return map( x -> x > 0 ? 1 / (1 + exp(- x)) : exp(x) / (1+exp(x)), x_vec)
end


# input is scalar
function inverse_logit_stability(x)
    if x > 0
        return 1 / (1 + exp(- x))
    else 
        expx = exp(x)
        return expx / (1 + expx)
    end
end


show(stdout, "text/plain", data_matrix)