#![feature(proc_macro)]
#![feature(use_extern_macros)]

#[macro_use]
extern crate stdweb;
#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate rand;
#[macro_use]
extern crate num_traits;
mod math;
use stdweb::js_export;
use stdweb::js_serializable;

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
    let probabilities = math::normalize_array(rate, &mut newvec);
    // roll n dice based on vec of probabilities
    let rolled_predictions = math::multinomial(1, &probabilities);
    let mut predicted_index = 0;
    // find the item with the most successful dice rolls
    let tuple_of_max = rolled_predictions.iter().enumerate().max_by(|&(_, item), &(_, y)| item.cmp(y));
    if let Some(max_tuple) = tuple_of_max {
        predicted_index = max_tuple.0;
    }
    //return to JS
    return Probsobj::new(predicted_index);
}
