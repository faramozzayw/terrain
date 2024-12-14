use bevy::{
    prelude::*,
    render::{
        mesh::{Indices, PrimitiveTopology},
        render_asset::RenderAssetUsages,
    },
};

use nalgebra::Vector3;

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

#[inline]
pub fn calculate_normal(
    vertex_0: Vector3<f32>,
    vertex_1: Vector3<f32>,
    vertex_2: Vector3<f32>,
) -> Vector3<f32> {
    let u = vertex_1 - vertex_0;
    let v = vertex_2 - vertex_0;

    u.cross(&v)
}

#[inline]
pub fn update_normals(
    (x, y, z): (usize, usize, usize),
    positions: &[Vector3<f32>],
    normals: &mut [Vector3<f32>],
) {
    let normal = calculate_normal(positions[x], positions[y], positions[z]);
    normals[x] += normal;
    normals[y] += normal;
    normals[z] += normal;
}

#[inline]
pub fn create_mesh(
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
) -> Mesh {
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));

    mesh
}
