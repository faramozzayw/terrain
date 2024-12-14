use crate::utils::{create_mesh, update_normals};
use bevy::prelude::*;
use nalgebra::Vector3;
use rayon::prelude::*;

pub struct Terrain {
    pub chunks: Vec<Chunk>,
}

#[derive(Debug, Clone)]
pub struct Chunk {
    positions: Vec<Vector3<f32>>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
    normals: Vec<Vector3<f32>>,
    pub chunk_x: usize,
    pub chunk_y: usize,
}

impl Terrain {
    pub const MAX_HEIGHT: f32 = 100.0;

    pub fn generate_chunks(heightmap: &[Vec<f32>], num_chunks: usize) -> Self {
        let mut chunks = Vec::with_capacity(num_chunks * num_chunks);

        for chunk_y in 0..num_chunks {
            for chunk_x in 0..num_chunks {
                let chunk = Chunk::generate(heightmap, chunk_x, chunk_y);

                chunks.push(chunk);
            }
        }

        Self { chunks }
    }
}

impl Chunk {
    pub const CHUNK_SIZE: usize = 128;

    pub fn into_mesh(mut self) -> Mesh {
        self.normals
            .par_iter_mut()
            .for_each(|normal| *normal = normal.normalize());

        let positions = self.positions.into_par_iter().map(Into::into).collect();
        let normals = self.normals.into_par_iter().map(Into::into).collect();

        create_mesh(positions, normals, self.uvs, self.indices)
    }

    pub fn generate(
        heightmap: &[Vec<f32>],
        original_chunk_x: usize,
        original_chunk_y: usize,
    ) -> Self {
        let chunk_x = original_chunk_x * Self::CHUNK_SIZE;
        let chunk_y = original_chunk_y * Self::CHUNK_SIZE;
        let mesh_size = Self::CHUNK_SIZE + 1;
        let size = mesh_size * mesh_size;

        let mut positions = Vec::with_capacity(size);
        let mut uvs = Vec::with_capacity(size);
        let mut indices = Vec::with_capacity((mesh_size - 1) * (mesh_size - 1) * 6);
        let mut normals = vec![Vector3::default(); size];

        let inv_size_minus_1 = 1.0 / (mesh_size - 1) as f32;

        for i in 0..mesh_size {
            for j in 0..mesh_size {
                let height = heightmap
                    .get(chunk_y + i)
                    .and_then(|row| row.get(chunk_x + j))
                    .cloned()
                    .unwrap_or(0.0)
                    * Terrain::MAX_HEIGHT;

                positions.push(Vector3::new(j as f32, height, i as f32));
                uvs.push([j as f32 * inv_size_minus_1, i as f32 * inv_size_minus_1]);
            }
        }

        for i in 0..mesh_size - 1 {
            for j in 0..mesh_size - 1 {
                let i0 = mesh_size * i + j;
                let i1 = i0 + 1;
                let i2 = mesh_size * (i + 1) + j;
                let i3 = i2 + 1;

                indices.extend_from_slice(&[
                    i0 as u32, i2 as u32, i1 as u32, i1 as u32, i2 as u32, i3 as u32,
                ]);

                update_normals((i0, i2, i1), &positions, &mut normals);
                update_normals((i1, i2, i3), &positions, &mut normals);
            }
        }

        Self {
            positions,
            normals,
            indices,
            uvs,
            chunk_x: original_chunk_x,
            chunk_y: original_chunk_y,
        }
    }
}
