data {
   int<lower=0> N;
   int<lower=1> D;
   vector[D] v1[N];
   vector[D] v2[N];
   real y[N];
 }
 parameters {
   vector[D] a;
   vector[D] s;
 }
 model {
 a~normal(0,1);  //multi_normal(0,I)
 s~normal(0,1);

 for(n in 1:N){

    vector[D] v1a;
    vector[D] v2a;
    vector[D] v1s;
    vector[D] v2s;
    real dot_a;
    real dot_s;

    v1a = v1[n].*a;
    v2a = v2[n].*a;
    dot_a = dot_product(v1a,v2a);
    dot_a = dot_a/sqrt(dot_product(v1a,v1a));
    dot_a = dot_a/sqrt(dot_product(v2a,v2a));
    dot_a = (dot_a+1)/2; // rescaling cosine similarity from [-1,1] to [0,1]

    v1s = v1[n].*s;
    v2s = v2[n].*s;
    dot_s = dot_product(v1s,v2s);
    dot_s = dot_s/sqrt(dot_product(v1s,v1s));
    dot_s = dot_s/sqrt(dot_product(v2s,v2s));
    dot_s = (dot_s+1)/2; // rescaling cosine similarity from [-1,1] to [0,1]

    target+= y[n]*beta_lpdf(dot_s|50,1) +(1-y[n])*beta_lpdf(dot_a|1,50);
 }
 }
