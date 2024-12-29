use bevy::prelude::*;
use nalgebra::Vector3;
use rayon::prelude::*;
use std::collections::HashMap;

use crate::utils::{create_mesh, update_normals};

pub struct Terrain {
    pub chunks: Vec<Chunk>,
}

type Normal = Vector3<f32>;
type Position = Vector3<f32>;
type UV = [f32; 2];

#[derive(Debug, Clone)]
pub struct Chunk {
    positions: Vec<Position>,
    uvs: Vec<UV>,
    indices: Vec<u32>,
    normals: Vec<Normal>,
    pub chunk_x: usize,
    pub chunk_y: usize,
}

impl Terrain {
    pub const MAX_HEIGHT: f32 = 100.0;

    pub fn generate_chunks(heightmap: &[Vec<f32>], num_chunks: usize) -> Self {
        let chunks = std::sync::Mutex::new(Vec::with_capacity(num_chunks * num_chunks));

        (0..num_chunks).into_par_iter().for_each(|chunk_y| {
            (0..num_chunks).into_par_iter().for_each(|chunk_x| {
                let chunk = Chunk::generate(heightmap, chunk_x, chunk_y);

                chunks.lock().unwrap().push(chunk);
            });
        });

        let mut terrain = Self {
            chunks: chunks.into_inner().unwrap(),
        };
        terrain.stitch_normals();
        terrain
    }

    pub fn stitch_normals(&mut self) {
        type VertexKey = (usize, usize);

        let capacity = self.chunks.len() * self.chunks[0].positions.len();

        // Collect normals from all chunks
        // For each chunk, we calculate the world-space position of each vertex using its local chunk position
        // and the chunk's global coordinates (chunk_x, chunk_y). This gives us a unique key for each vertex.
        // If a vertex appears in multiple chunks (i.e., it's on the edge of a chunk), its world-space position
        // will be the same, and the normals from all chunks sharing this vertex will be averaged.
        let normal_map: HashMap<VertexKey, Vec<Normal>> = self
            .chunks
            .par_iter()
            .map(|chunk| {
                let mut normals: HashMap<VertexKey, Vec<Normal>> =
                    HashMap::with_capacity(chunk.positions.len());

                chunk
                    .positions
                    .iter()
                    .enumerate()
                    .for_each(|(i, position)| {
                        let pos = chunk.get_world_space_position(position);
                        normals.entry(pos).or_default().push(chunk.normals[i]);
                    });

                normals
            })
            .reduce(
                || HashMap::with_capacity(capacity),
                |mut acc, local_map| {
                    for (key, normals) in local_map {
                        acc.entry(key).or_default().extend(normals);
                    }
                    acc
                },
            );

        // If the vertex is not shared (i.e., it only exists in one chunk), no averaging happens for it; the normal stays the same.
        // If the vertex is shared (i.e., it exists in multiple chunks), the normals are averaged.
        let averaged_normals: HashMap<VertexKey, Normal> = normal_map
            .par_iter()
            .map(|(key, normals)| {
                let average = normals
                    .iter()
                    .fold(Normal::zeros(), |acc, n| acc + *n)
                    .normalize();
                (*key, average)
            })
            .collect();

        self.chunks.par_iter_mut().for_each(|chunk| {
            chunk.normals = chunk
                .positions
                .par_iter()
                .map(|position| {
                    let pos = chunk.get_world_space_position(position);
                    averaged_normals[&pos]
                })
                .collect();
        });
    }
}

impl Chunk {
    pub const CHUNK_SIZE: usize = 128;

    #[inline]
    pub fn get_world_space_position(&self, position: &Position) -> (usize, usize) {
        let x = self.chunk_x * Chunk::CHUNK_SIZE + position.x as usize;
        let y = self.chunk_y * Chunk::CHUNK_SIZE + position.z as usize;

        (x, y)
    }

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

        let inv_size_minus_1 = 1.0 / (mesh_size - 1) as f32;

        let (positions, uvs): (Vec<_>, Vec<_>) = (0..mesh_size)
            .flat_map(|i| (0..mesh_size).map(move |j| (i, j)))
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(i, j)| {
                let height = heightmap
                    .get(chunk_y + i)
                    .and_then(|row| row.get(chunk_x + j))
                    .cloned()
                    .unwrap_or(0.0)
                    * Terrain::MAX_HEIGHT;

                (
                    Vector3::new(j as f32, height, i as f32),
                    [j as f32 * inv_size_minus_1, i as f32 * inv_size_minus_1],
                )
            })
            .unzip();

        let normals = std::sync::Mutex::new(vec![Normal::default(); size]);

        let indices: Vec<u32> = (0..mesh_size - 1)
            .flat_map(|i| (0..mesh_size - 1).map(move |j| (i, j)))
            .collect::<Vec<_>>()
            .into_par_iter()
            .flat_map(|(i, j)| {
                let i0 = mesh_size * i + j;
                let i1 = i0 + 1;
                let i2 = mesh_size * (i + 1) + j;
                let i3 = i2 + 1;

                let mut normals = normals.lock().unwrap();
                update_normals((i0, i2, i1), &positions, &mut normals);
                update_normals((i1, i2, i3), &positions, &mut normals);

                vec![
                    i0 as u32, i2 as u32, i1 as u32, // First triangle
                    i1 as u32, i2 as u32, i3 as u32, // Second triangle
                ]
            })
            .collect();

        let normals = normals.into_inner().unwrap();

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

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    fn create_mock_chunk(
        positions: Vec<Vector3<f32>>,
        normals: Vec<Vector3<f32>>,
        chunk_x: usize,
        chunk_y: usize,
    ) -> Chunk {
        let uvs = vec![[0.0, 0.0]; positions.len()]; // Dummy UVs
        let indices = vec![]; // Dummy indices, not needed for this test
        Chunk {
            positions,
            normals,
            uvs,
            indices,
            chunk_x,
            chunk_y,
        }
    }

    #[test]
    fn test_stitch_normals() {
        let positions_chunk1 = vec![
            Vector3::new(0.0, 1.0, 0.0), // Vertex at (0, 0)
            Vector3::new(1.0, 1.0, 0.0), // Vertex at (1, 0)
            Vector3::new(0.0, 1.0, 1.0), // Vertex at (0, 1)
        ];
        let normals_chunk1 = vec![
            Vector3::new(0.0, 1.0, 0.0), // Normal for vertex (0, 0)
            Vector3::new(0.0, 1.0, 0.0), // Normal for vertex (1, 0)
            Vector3::new(0.0, 1.0, 0.0), // Normal for vertex (0, 1)
        ];

        let positions_chunk2 = vec![
            Vector3::new(1.0, 2.0, 0.0), // Vertex at (1, 0)
            Vector3::new(2.0, 2.0, 0.0), // Vertex at (2, 0)
            Vector3::new(1.0, 2.0, 1.0), // Vertex at (1, 1)
        ];
        let normals_chunk2 = vec![
            Vector3::new(0.0, 1.0, 0.0), // Normal for vertex (1, 0)
            Vector3::new(0.0, 1.0, 0.0), // Normal for vertex (2, 0)
            Vector3::new(0.0, 1.0, 0.0), // Normal for vertex (1, 1)
        ];

        let chunk1 = create_mock_chunk(positions_chunk1, normals_chunk1, 0, 0);
        let chunk2 = create_mock_chunk(positions_chunk2, normals_chunk2, 1, 0);

        let mut terrain = Terrain {
            chunks: vec![chunk1, chunk2],
        };

        terrain.stitch_normals();

        // Check the normals for shared vertices
        let chunk1_normals = &terrain.chunks[0].normals;
        let chunk2_normals = &terrain.chunks[1].normals;

        assert_eq!(chunk1_normals[1], Vector3::new(0.0, 1.0, 0.0)); // Vertex (1, 0) in chunk1
        assert_eq!(chunk2_normals[0], Vector3::new(0.0, 1.0, 0.0)); // Vertex (1, 0) in chunk2

        // For vertex (0, 1) in chunk1, it should remain the same
        assert_eq!(chunk1_normals[2], Vector3::new(0.0, 1.0, 0.0)); // Vertex (0, 1) in chunk1

        // For vertex (2, 0) in chunk2, it should remain the same
        assert_eq!(chunk2_normals[1], Vector3::new(0.0, 1.0, 0.0)); // Vertex (2, 0) in chunk2
    }
}
