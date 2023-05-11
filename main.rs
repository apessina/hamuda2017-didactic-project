// Hamuda et al. (2017) crop detection algorithm adaptation for Rust didactic project. DOI: https://doi.org/10.1016/j.compag.2016.11.021
// André L. R. Pessina

// import crates
use opencv::{
    core,
    imgcodecs,
    imgproc,
    types,
    Result,
};

fn main() -> Result<()> {

    // Load the image
    let mut img = imgcodecs::imread("./src/input.jpg", imgcodecs::IMREAD_COLOR)?;

    // Apply Gaussian Blur filter
    let mut blurred_img = core::Mat::default();
    imgproc::gaussian_blur(
        &img,
        &mut blurred_img,
        core::Size::new(3, 3),
        0.0,
        0.0,
        core::BORDER_DEFAULT,
    )?;
    // Save the blurred image
    imgcodecs::imwrite("./src/gauss_output.jpg", &blurred_img, &types::VectorOfi32::new())?;

    // Convert to HSV color space
    let mut hsv_img = core::Mat::default();
    imgproc::cvt_color(&blurred_img, &mut hsv_img, imgproc::COLOR_BGR2HSV, 0)?;
    // Save the HSV image
    imgcodecs::imwrite("./src/hsv_output.jpg", &hsv_img, &types::VectorOfi32::new())?;

    // Apply thresholding to the HSV channels
    let mut filtered_bin_img = core::Mat::default();
    let min_hsv = core::Scalar::new(65.0, 33.0, 117.0, 0.0);
    let max_hsv = core::Scalar::new(98.0, 255.0, 255.0, 0.0);
    core::in_range(&hsv_img, &min_hsv, &max_hsv, &mut filtered_bin_img)?;
    // Save the filtered binary image
    imgcodecs::imwrite("./src/filtered_output.jpg", &filtered_bin_img, &core::Vector::new())?;

    // Apply morphological erosion
    let mut eroded_img = core::Mat::default();
    //// Create a 7x7 structuring element for dilation
    let erode_kernel = imgproc::get_structuring_element( // create a 3x3 rectangular structuring element
        0, // MORPH_RECT
        core::Size::new(3, 3),
        core::Point::new(-1, -1),
    ).unwrap();
    imgproc::erode(&filtered_bin_img, &mut eroded_img, &erode_kernel, core::Point::default(), 1, core::BORDER_CONSTANT, core::Scalar::default())?; // imgproc::morphology_default_border_value; std anchor (-1, -1)
    // Apply morphological dilation
    let mut dilated_img = core::Mat::default();
    let dilate_kernel = imgproc::get_structuring_element( // create a 7x7 rectangular structuring element
        0, // MORPH_RECT
        core::Size::new(7, 7),
        core::Point::new(-1, -1),
    ).unwrap();
    imgproc::dilate(&eroded_img, &mut dilated_img, &dilate_kernel, core::Point::default(), 1, core::BORDER_CONSTANT, core::Scalar::default())?; // imgproc::morphology_default_border_value; std anchor (-1, -1)
    // Save the result as a binary black and white image
    imgcodecs::imwrite("./src/morpho_output.jpg", &dilated_img, &core::Vector::new())?;

    // Find objects contours
    let mut contours = types::VectorOfVectorOfPoint::new();
    let mut hierarchy = core::Mat::default();
    imgproc::find_contours_with_hierarchy(
        &dilated_img,
        &mut contours,
        &mut hierarchy,
        imgproc::RETR_EXTERNAL,
        imgproc::CHAIN_APPROX_SIMPLE,
        core::Point::new(0, 0), // Point::default()
    )?;

    // Draw the contours on the original image
    let mut img_contours = img.clone();
    imgproc::draw_contours(&mut img_contours, &contours, -1, core::Scalar::new(0.0, 255.0, 0.0, 0.0), -1, 8, &mut hierarchy, 2, core::Point::default())?;
    // Save contours in output image
    imgcodecs::imwrite("./src/count_output.jpg", &img_contours, &core::Vector::new())?;

    // Loop over all the contours and calculate their area and perimeter
    let mut areas = vec![]; // store selected areas
    for contour in contours {
        let area = imgproc::contour_area(&contour, false)?;
        let perimeter = imgproc::arc_length(&contour, true)?;

        // Check if the area and perimeter meet the detection rule criteria
        if area >= 600.0 && perimeter >= 30.0 {

            // Draw a red rectangle around the contour
            let rect = imgproc::bounding_rect(&contour)?;
            imgproc::rectangle(
                &mut img,
                rect,
                core::Scalar::new(0.0, 0.0, 255.0, 0.0),
                2,
                imgproc::LINE_8,
                0,
            )?;

            // Save the image with the contour and rectangle drawn
            imgcodecs::imwrite("./src/final_output.jpg", &img, &core::Vector::new())?;
        
            // Store area for later use
            areas.push(area);

        }

    }

    println!("areas: {:#?}", &mut areas);

    Ok(())

}
