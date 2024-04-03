mod data;
mod model;
mod training;
mod inference;

use crate::model::ModelConfig;
use burn::backend::{wgpu::AutoGraphicsApi, Autodiff, Wgpu};
use burn::data::dataset::vision::MnistItem;
use burn::optim::AdamConfig;
use image::io::Reader as ImageReader;
use image::GrayImage;

fn main() {
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();

    //TRAIN THE MODEL
    // crate::training::train::<MyAutodiffBackend>(
    //     //"./tmp/my_first_burn_model",
    //     "./tmp/my_first_burn_model",
    //     crate::training::TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
    //     device,
    // );

    //INFER THE MODEL
    // let item_to_infer = MnistItem {
    //     image: [[0.0; 28]; 28], // Fake image data
    //     label: 0, // Fake label
    // };
    let img = ImageReader::open("./assets_to_run_inference/testSet/img_11.jpg")
        .unwrap()
        .decode()
        .unwrap()
        .to_luma8();

    let img_array: [[f32; 28]; 28] = img.into_raw().chunks(28).enumerate().map(|(i, row)| {
        let mut arr: [f32; 28] = [0.0; 28];
        for (j, pixel) in row.iter().enumerate() {
            arr[j] = *pixel as f32 / 255.0; // normalize pixel value to [0, 1]
        }
        arr
    }).collect::<Vec<_>>().try_into().unwrap();
    println!("{:?}", &img_array);

    let item_to_infer = MnistItem {
        image: img_array,
        label: 2, // replace with actual label if available
    };

    crate::inference::infer::<MyAutodiffBackend>(
        "./tmp/my_first_burn_model",
        device,
        item_to_infer
    );
}
