data {
   int<lower=0> N;
   int<lower=1> D;
   vector[D] v1[N];
   vector[D] v2[N];
   real y[N];
 }
 parameters {
   vector[D] a;
 }
 model {
 a~normal(0,1);  //multi_normal(0,I);
 
 for(n in 1:N){

    vector[D] v1t;
    vector[D] v2t;
    real dot;
    
    v1t = v1[n].*a;
    v2t = v2[n].*a;
    dot = dot_product(v1t,v2t);
    dot = dot/sqrt(dot_product(v1t,v1t));
    dot = dot/sqrt(dot_product(v2t,v2t));

    dot = (dot+1)/2; // rescaling cosine similarity from [-1,1] to [0,1]

    target+= y[n]*beta_lpdf(dot|50,1) +(1-y[n])*beta_lpdf(dot|1,50);
 }
 }
