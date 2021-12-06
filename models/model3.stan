data {
   int<lower=0> N;
   int<lower=1> D;
   vector[D] v1[N];
   vector[D] v2[N];
   real y[N];
 }
 parameters {
   vector[D] a[D];
 }
 model {

 for(i in 1:D){
 a[i]~normal(0,1);  //multi_normal(0,I);
 }
 
 for(n in 1:N){

    vector[D] v1t;
    vector[D] v2t;
    real dot;

    for(j in 1:D){
    v1t[j]=dot_product(v1[n],a[j]);
    v2t[j]=dot_product(v2[n],a[j]);
    }

    dot = dot_product(v1t,v2t);
    dot = dot/sqrt(dot_product(v1t,v1t));
    dot = dot/sqrt(dot_product(v2t,v2t));

    dot = (dot+1)/2; // rescaling cosine similarity from [-1,1] to [0,1]

    target+= y[n]*beta_lpdf(dot|25,1) +(1-y[n])*beta_lpdf(dot|1,25);
 }
 }
