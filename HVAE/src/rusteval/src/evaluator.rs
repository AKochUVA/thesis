use pyo3::prelude::*;

use pyo3::exceptions::PyValueError;
use ndarray::Array1;
use std::collections::HashMap;
// use rayon::prelude::*;

#[pyclass]
#[derive(Clone, Debug)]
pub struct Evaluator {
    data: HashMap<String, Array1<f64>>, // HashMap to store variable data
    true_val: Array1<f64>,              // Array to store true target values
    var_len: usize,                     // Length of the variables
    classification: bool,               // boolean to determine if classification mode is enabled
    threshold: f64                      // Threshold value for classification
}

impl Evaluator {
   fn _eval_expr(&self, expr: &Vec<&str>, constants: Vec<f64>, safe_operations: bool) -> PyResult<Array1<f64>>{
       let mut cns = constants.into_iter();
       let mut stack: Vec<Array1<f64>> = Vec::new();
        for &t in expr {
            match t {
                "+" => {
                    let val = stack.pop().ok_or(PyValueError::new_err("Exception during evaluation: Ran out of tokens in the stack."))? + stack.pop().ok_or(PyValueError::new_err("Exception during evaluation: Ran out of tokens in the stack."))?;
                    stack.push(val)
                }
                "-" => {
                    let b = stack.pop().ok_or(PyValueError::new_err("Exception during evaluation: Ran out of tokens in the stack."))?;
                    let a = stack.pop().ok_or(PyValueError::new_err("Exception during evaluation: Ran out of tokens in the stack."))?;
                    stack.push(a-b)
                }
                "*" => {
                    let a = stack.pop().ok_or(PyValueError::new_err("Exception during evaluation: Ran out of tokens in the stack."))? * stack.pop().ok_or(PyValueError::new_err("Exception during evaluation: Ran out of tokens in the stack."))?;
                    stack.push(a)
                }
                "/" => {
                    let b = stack.pop().ok_or(PyValueError::new_err("Exception during evaluation: Ran out of tokens in the stack."))?;
                    let a = stack.pop().ok_or(PyValueError::new_err("Exception during evaluation: Ran out of tokens in the stack."))? / b;
                    if safe_operations {
                       if a.iter().all(|v| v.is_finite()){
                           stack.push(a);
                       } else {
                           return Err(PyValueError::new_err("Exception during evaluation: Division by zero."))
                       }
                    } else {
                        stack.push(a);
                    }
                }
                "log" => {
                    let a = stack.pop().ok_or(PyValueError::new_err("Exception during evaluation: Ran out of tokens in the stack."))?;
                    let b = a.map(|v| v.ln());
                    if safe_operations {
                        if b.iter().all(|v| v.is_finite()) {
                            stack.push(b);
                        } else {
                            return Err(PyValueError::new_err("Exception during evaluation: Log of a negative number."))
                        }
                    } else {
                        stack.push(b);
                    }
                }
                "sqrt" => {
                    let a = stack.pop().ok_or(PyValueError::new_err("Exception during evaluation: Ran out of tokens in the stack."))?;
                    let b = a.map(|v| v.sqrt());
                    if safe_operations {
                        if b.iter().all(|v| v.is_finite()) {
                            stack.push(b);
                        } else {
                            return Err(PyValueError::new_err("Exception during evaluation: Square root of a negative number."))
                        }
                    } else {
                        stack.push(b);
                    }
                }
                "cos" => {
                    let a = stack.pop().ok_or(PyValueError::new_err("Exception during evaluation: Ran out of tokens in the stack."))?;
                    stack.push(a.map(|v| v.cos()));
                }
                "sin" => {
                    let a = stack.pop().ok_or(PyValueError::new_err("Exception during evaluation: Ran out of tokens in the stack."))?;
                    stack.push(a.map(|v| v.sin()));
                }
                "exp" => {
                    let a = stack.pop().ok_or(PyValueError::new_err("Exception during evaluation: Ran out of tokens in the stack."))?;
                    stack.push(a.map(|v| v.exp()));
                }
                "^2" => {
                    let a = stack.pop().ok_or(PyValueError::new_err("Exception during evaluation: Ran out of tokens in the stack."))?;
                    stack.push(a.map(|v| v.powi(2)));
                }
                "^3" => {
                    let a = stack.pop().ok_or(PyValueError::new_err("Exception during evaluation: Ran out of tokens in the stack."))?;
                    stack.push(a.map(|v| v.powi(3)));
                }
                "^4" => {
                    let a = stack.pop().ok_or(PyValueError::new_err("Exception during evaluation: Ran out of tokens in the stack."))?;
                    stack.push(a.map(|v| v.powi(4)));
                }
                "^5" => {
                    let a = stack.pop().ok_or(PyValueError::new_err("Exception during evaluation: Ran out of tokens in the stack."))?;
                    stack.push(a.map(|v| v.powi(5)));
                }
                "C" => {
                    let c = cns.next().ok_or(PyValueError::new_err("Exception during evaluation: Not enough constants were given."))?;
                    stack.push(Array1::<f64>::zeros(self.var_len).map(|x| x+c));
                }
                _ => {
                    if self.data.contains_key(t) {
                        let variable = self.data.get(t).unwrap().clone();
                        stack.push(variable);
                    }
                    else {
                        match t.parse::<f64>() {
                            Ok(v) => stack.push(Array1::<f64>::zeros(self.var_len).map(|x| x+v)),
                            Err(_) => return Err(PyValueError::new_err(format!("Exception during evaluation: Token {} not found, check tokens or the rusteval/src/lib.rs file.", t)))
                        }
                    }
                }
            }
        }
       // Old method (regression only)
       //  if stack.len() == 1 {
       //      return Ok(stack.pop().unwrap());
       //  } else {
       //      return Err(PyValueError::new_err("Exception during evaluation: More than one value in the evaluation stack at the end (some tokens might be missing)."))
       //  }

       // New method (including sigmoid function)
       if stack.len() == 1 {
            let result = stack.pop().unwrap();
            if self.classification {
                //let sigmoid_result = result.map(|v| 1.0 / (1.0 + (-v).exp()));
                //Ok(sigmoid_result.map(|&v| if v >= self.threshold { 1.0 } else { 0.0 }))
                Ok(result.map(|v| 1.0 / (1.0 + (-v).exp())))
            } else {
                Ok(result)
            }
        } else {
            Err(PyValueError::new_err("Exception during evaluation: More than one value in the stack at the end (some tokens might be missing)."))
        }
   }
}

#[pymethods]
impl Evaluator {
    #[new]
    pub fn new(data: Vec<Vec<f64>>, names: Vec<String>, target: Vec<f64>, classification: bool, threshold: f64) -> PyResult<Self> {
        if data.len() != names.len() {
            return Err(PyValueError::new_err("Exception during initialization of the Evaluator: Arguments data and names are not of the same length."))
        }
        let var_len = if data.len() > 0 {data.get(0).unwrap().len()} else {1};

        if target.len() != var_len {
            return Err(PyValueError::new_err("Exception during initialization of the Evaluator: Target and data vectors are not of the same length."))
        }

        if threshold < 0.0 || threshold > 1.0 {
            return Err(PyValueError::new_err("Exception during initialization of the Evaluator: Threshold value must be between 0 and 1."))
        }

        let mut hm = HashMap::new();
        for (d, n) in data.into_iter().zip(names) {
            hm.insert(n, Array1::from_vec(d));
        }

        let true_val = Array1::from_vec(target);
        return Ok(Evaluator { data: hm, var_len, true_val, classification, threshold})
    }

    fn eval_expr(&self, expr: Vec<&str>, constants: Vec<Vec<f64>>, safe_operations: bool) -> PyResult<Vec<Vec<f64>>>{
        return constants.into_iter()
            .map(|cst| match self._eval_expr(&expr, cst, safe_operations) {
                Ok(v) => Ok(v.into_raw_vec()),
                Err(x) => Err(x)
            }).collect();
    }


    pub fn get_rmse(&self, expr: Vec<&str>, constants: Vec<Vec<f64>>, default_value: f64, verbose: bool) -> PyResult<Vec<f64>>{
        let errors = constants.into_iter()
            .map(|cns| match self._eval_expr(&expr, cns, true) {
                Err(x) => {
                    if verbose {
                        println!("{}", x.to_string());
                    }
                    default_value
                },
                Ok(pv) => {
                    let s: f64 = (self.true_val.clone() - pv)
                    .into_iter()
                    .map(|v| f64::powi(v, 2))
                    .sum();
                    let rmse: f64 = (s / (self.true_val.len() as f64)).sqrt();
                    if rmse.is_finite(){
                        rmse
                    }
                    else {
                        if verbose {
                            println!("Exception during evaluation: Overflow when calculating rmse.");
                        }
                        default_value
                    }
                }
            }).collect();
        Ok(errors)
    }

    pub fn get_cross_entropy_loss(&self, expr: Vec<&str>, constants: Vec<Vec<f64>>, default_value: f64, verbose: bool) -> PyResult<Vec<f64>>{
        let errors = constants.into_iter()
            .map(|cns| match self._eval_expr(&expr, cns, true) {
                Err(x) => {
                    if verbose {
                        println!("{}", x.to_string());
                    }
                    default_value
                },
                Ok(pv) => {
                    let cross_entropy: f64 = (&self.true_val * &pv.mapv(f64::ln) + (&self.true_val.mapv(|v| 1.0 - v) * &pv.mapv(|v| (1.0 - v).ln())))
                        .sum() * -1.0 / self.true_val.len() as f64;

                    print!("Cross entropy loss: ");
                    println!("{}", cross_entropy.to_string());

                    if cross_entropy.is_finite() {
                        cross_entropy
                    } else {
                        if verbose {
                            println!("Exception during evaluation: Overflow when calculating cross-entropy loss.");
                        }
                        default_value
                    }
                }
            }).collect();
        Ok(errors)
    }


    // pub fn get_cross_entropy_loss(&self, expr: Vec<&str>, constants: Vec<Vec<f64>>, default_value: f64, verbose: bool) -> PyResult<Vec<f64>> {
    //     constants.into_iter()
    //         .map(|cns| {
    //             match self._eval_expr(&expr, cns, true) {
    //                 Ok(pv) => {
    //                     let cross_entropy: f64 = (&self.true_val * &pv.mapv(f64::ln) + (&self.true_val.mapv(|v| 1.0 - v) * &pv.mapv(|v| (1.0 - v).ln())))
    //                         .sum() * -1.0 / self.true_val.len() as f64;
    //                     if cross_entropy.is_finite() {
    //                         cross_entropy
    //                     } else {
    //                         if verbose {
    //                             println!("Exception during evaluation: Overflow when calculating cross-entropy loss.");
    //                         }
    //                         default_value
    //                     }
    //                 }
    //                 Err(e) => {
    //                     if verbose {
    //                         println!("{}", e);
    //                     }
    //                     default_value
    //                 }
    //             }
    //         })
    //         .collect()
    // }
}