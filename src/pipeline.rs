// use std::collections::HashMap;

// pub struct Pipeline {
//     steps: HashMap<String, fn() -> ()>,
// }

// impl Pipeline {
//     pub fn pipeline(steps: HashMap<String, fn() -> ()>) -> Self {
//         Pipeline {
//             steps: steps,
//         }
//     }

//     pub fn print_map(&self) {
//         for (key, value) in &self.steps {
//             println!("{}", key);
//             value()
//         }
//     }
// }
