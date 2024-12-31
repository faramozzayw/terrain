pub fn parse_heightmap(path: &str) -> Vec<Vec<f32>> {
    let img = image::open(path)
        .expect("Failed to open heightmap")
        .into_luma8(); // Convert to grayscale

    let (width, height) = img.dimensions();
    assert_eq!(width, height, "Heightmap must be NxN");

    let size = width as usize;
    let mut heightmap = vec![vec![0.0; size]; size];

    for (i, row) in heightmap.iter_mut().enumerate().take(size) {
        for (j, v) in row.iter_mut().enumerate().take(size) {
            // Grayscale intensity & normalize [0.0, 1.0]
            *v = img.get_pixel(j as u32, i as u32)[0] as f32 / 255.0;
        }
    }

    heightmap
}
