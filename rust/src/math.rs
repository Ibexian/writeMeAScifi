use super::num_traits::Float;
use super::rand::distributions::{Binomial, Distribution};
use super::rand::{RngCore, ChaChaRng};

pub fn normalize_array(rate: f64,  arr: &mut Vec<f64> ) -> Vec<f64> {
    //Take the input vec
    let mod_array = arr.iter()
    //log of each
        .map(|x| x.log(super::std::f64::consts::E))
    //divide each by rate
        .map(|x| x / rate )
    //exp each
        .map(|x| x.exp() )
        .collect::<Vec<f64>>();
    //sum up
    let sum = sum(&mod_array);
    //divide each by sum to normalize
    let normalized_array = mod_array.iter()
        .map(|x| x / sum)
        .collect::<Vec<f64>>();

    return normalized_array
}

fn sum ( arr: &Vec<f64> ) -> f64 {
    arr.iter()
        .fold(0.0, |total, next| total + next)
}

pub fn multinomial (rolls: u64, probs: &Vec<f64>) -> Vec<u64> {
    //Take Vector of probabilities
    //Return Vector of results - for n rolls

    let d = probs.len();
    let mut results = vec![0;d];
    let mut sum;
    let mut i = 0;
    let rand_seed = 1234;

    while i < d {
        sum = 1.0;
        let mut dn = rolls; //# of dice to roll

        for j in 0..d {
            //rand::thread_rng panics in WASM, so we use ChaChaRng instead
            results[i+j] = Binomial::new(dn, probs[j]/sum).sample(&mut ChaChaRng::new_unseeded());
            dn = dn - results[i+j];
            if dn <= 0 {
                break
            }
            sum = sum - probs[j];
        }
        if dn > 0 {
            results[i+d-1] = dn;
        }

        i = i + d;

    }

    return results
}
