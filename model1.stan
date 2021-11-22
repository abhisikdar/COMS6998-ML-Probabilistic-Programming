data {
   int<lower=0> N;
   int<lower=1> D;
   vector[D] v1[N];
   vector[D] v2[N];
   real y[N];
 }
 parameters {
   vector[D] a;
   real samp1;
   real samp2;
 }
 model {

 for(n in 1:N){
    a~normal(0,1);  //multi_normal(rep_vector(0,D),);

    vector[D] v1t;
    vector[D] v2t;
    real dot;
    //real samp1;
    //real samp2;

    v1t = v1[n].*a;
    v2t = v2[n].*a;
    dot = dot_product(v1t,v2t);
    dot = dot/sqrt(dot_product(v1t,v1t));
    dot = dot/sqrt(dot_product(v2t,v2t));

    dot = (dot+1)/2; // rescaling cosine similarity from [-1,1] to [0,1]

    samp1~beta(1+dot,1-dot);
    samp2~beta(1-dot,1+dot);

    target+= y[n]*beta_lpdf(samp1|50,1) +(1-y[n])*beta_lpdf(samp2|1,50);
 }
 }
