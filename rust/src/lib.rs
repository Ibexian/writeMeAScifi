#![feature(proc_macro)]
#![feature(use_extern_macros)]

#[macro_use]
extern crate stdweb;
#[macro_use]
extern crate num_traits;
#[macro_use]
extern crate serde_derive;

use stdweb::js_export;
use stdweb::js_serializable;
use num_traits::Float;

#[derive(Serialize, Deserialize)]
struct probsObj {
    result: Vec<f64>
}

impl probsObj {
    fn new(arr: Vec<f64>) -> Self {
        probsObj {
            result: arr
        }
    }
}

js_serializable!(probsObj);

#[js_export]
fn sample (rate: f64, vec: Vec<f64>) -> probsObj {
    //borrows the vector from JS
    //Does the sampling work and returns the result
    let mut newvec = vec.to_vec();
    let probabilities = normalize_array(rate, &mut newvec);
    return probsObj::new(probabilities)
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

// fn multinomial (probs: &Vec<f64>) {
//     //Create an array of probabilities
//     // let probabilities = sampling.Multinomial(1, preds.tolist(), 1);
//     // //Return the one selected id based on probs
//     // return probabilities.draw().reduce((iMax, x, i, arr) => x > arr[iMax] ? i : iMax, 0);
// }
