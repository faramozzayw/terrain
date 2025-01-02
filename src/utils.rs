use bevy::prelude::*;
use bevy::render::mesh::VertexAttributeValues;

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

pub fn get_mut_position_from_mesh(chunk_mesh: &mut Mesh) -> Option<&mut Vec<[f32; 3]>> {
    match chunk_mesh.attribute_mut(Mesh::ATTRIBUTE_POSITION)? {
        VertexAttributeValues::Float32x3(positions) => Some(positions),
        _ => None,
    }
}
