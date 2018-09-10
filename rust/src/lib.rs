#![feature(proc_macro)]
#![feature(use_extern_macros)]

#[macro_use]
extern crate stdweb;
#[macro_use]
extern crate num_traits;
#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate rand;
extern crate juggernaut;

use stdweb::js_export;
use stdweb::js_serializable;
use num_traits::Float;
use rand::distributions::{Binomial, Distribution};
use rand::{RngCore, ChaChaRng};

#[derive(Serialize, Deserialize)]
struct Probsobj {
    result: usize
}

impl Probsobj {
    fn new(max_index: usize) -> Self {
        Probsobj {
            result: max_index
        }
    }
}

js_serializable!(Probsobj);

#[js_export]
fn sample (rate: f64, vec: Vec<f64>) -> Probsobj {
    //borrows the vector from JS
    //Does the sampling work and returns the result
    let mut newvec = vec.to_vec();
    // create a normalized vec of probabilities
    let probabilities = normalize_array(rate, &mut newvec);
    // roll n dice based on vec of probabilities
    let rolled_predictions = multinomial(1, &probabilities);
    let mut predicted_index = 0;
    // find the item with the most successful dice rolls
    let tuple_of_max = rolled_predictions.iter().enumerate().max_by(|&(_, item), &(_, y)| item.cmp(y));
    if let Some(max_tuple) = tuple_of_max {
        predicted_index = max_tuple.0;
    }
    //return to JS
    return Probsobj::new(predicted_index);
}


fn normalize_array(rate: f64,  arr: &mut Vec<f64> ) -> Vec<f64> {
    //Take the input vec
    let mod_array = arr.iter()
    //log of each
        .map(|x| x.log(std::f64::consts::E))
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

fn multinomial (rolls: u64, probs: &Vec<f64>) -> Vec<u64> {
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
