use bevy::{
    prelude::*,
    render::{
        mesh::{Indices, PrimitiveTopology},
        render_asset::RenderAssetUsages,
    },
};
use rayon::prelude::*;
use std::collections::HashMap;

pub struct Terrain {
    pub chunks: Vec<Chunk>,
}

type Normal = Vec3;
type Position = Vec3;
type UV = [f32; 2];

#[derive(Debug, Clone)]
pub struct Chunk {
    pub positions: Vec<Position>,
    pub uvs: Vec<UV>,
    pub indices: Vec<usize>,
    pub normals: Vec<Normal>,
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
                    .fold(Normal::ZERO, |acc, n| acc + *n)
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
    pub const SIZE: usize = 128;

    #[inline]
    pub fn face_normal(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
        let u = b - a;
        let v = c - a;

        u.cross(v)
    }

    #[inline]
    pub fn get_world_space_position(&self, position: &Position) -> (usize, usize) {
        let x = self.chunk_x * Chunk::SIZE + position.x as usize;
        let y = self.chunk_y * Chunk::SIZE + position.z as usize;

        (x, y)
    }

    pub fn into_mesh(mut self) -> Mesh {
        self.normals
            .par_iter_mut()
            .for_each(|normal| *normal = normal.normalize());

        let positions = self.positions.into_par_iter().map(Into::into).collect();
        let normals = self.normals.into_par_iter().map(Into::into).collect();
        let indices = self.indices.into_par_iter().map(|v| v as u32).collect();

        let mut mesh = Self::create_mesh(positions, normals, self.uvs, indices);
        // TODO: optimize this
        mesh.generate_tangents().unwrap();
        mesh
    }

    #[inline]
    fn create_mesh(
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

    pub fn generate(
        heightmap: &[Vec<f32>],
        original_chunk_x: usize,
        original_chunk_y: usize,
    ) -> Self {
        let chunk_x = original_chunk_x * Self::SIZE;
        let chunk_y = original_chunk_y * Self::SIZE;
        let mesh_size = Self::SIZE + 1;
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
                    Vec3::new(j as f32, height, i as f32),
                    [j as f32 * inv_size_minus_1, i as f32 * inv_size_minus_1],
                )
            })
            .unzip();

        let indices: Vec<_> = (0..(mesh_size - 1) * (mesh_size - 1))
            .into_par_iter()
            .flat_map(|index| {
                let i = index / (mesh_size - 1);
                let j = index % (mesh_size - 1);

                let i0 = mesh_size * i + j;
                let i1 = i0 + 1;
                let i2 = mesh_size * (i + 1) + j;
                let i3 = i2 + 1;

                [i0, i2, i1, i1, i2, i3] // First and second triangles
            })
            .collect();

        let mut normals = vec![Normal::default(); size];

        indices.chunks_exact(3).for_each(|indices| {
            let (a, b, c) = (indices[0], indices[1], indices[2]);
            let normal = Self::face_normal(positions[a], positions[b], positions[c]);
            normals[a] += normal;
            normals[b] += normal;
            normals[c] += normal;
        });

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

    fn create_mock_chunk(
        positions: Vec<Vec3>,
        normals: Vec<Vec3>,
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
            Vec3::new(0.0, 1.0, 0.0), // Vertex at (0, 0)
            Vec3::new(1.0, 1.0, 0.0), // Vertex at (1, 0)
            Vec3::new(0.0, 1.0, 1.0), // Vertex at (0, 1)
        ];
        let normals_chunk1 = vec![
            Vec3::new(0.0, 1.0, 0.0), // Normal for vertex (0, 0)
            Vec3::new(0.0, 1.0, 0.0), // Normal for vertex (1, 0)
            Vec3::new(0.0, 1.0, 0.0), // Normal for vertex (0, 1)
        ];

        let positions_chunk2 = vec![
            Vec3::new(1.0, 2.0, 0.0), // Vertex at (1, 0)
            Vec3::new(2.0, 2.0, 0.0), // Vertex at (2, 0)
            Vec3::new(1.0, 2.0, 1.0), // Vertex at (1, 1)
        ];
        let normals_chunk2 = vec![
            Vec3::new(0.0, 1.0, 0.0), // Normal for vertex (1, 0)
            Vec3::new(0.0, 1.0, 0.0), // Normal for vertex (2, 0)
            Vec3::new(0.0, 1.0, 0.0), // Normal for vertex (1, 1)
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

        assert_eq!(chunk1_normals[1], Vec3::new(0.0, 1.0, 0.0)); // Vertex (1, 0) in chunk1
        assert_eq!(chunk2_normals[0], Vec3::new(0.0, 1.0, 0.0)); // Vertex (1, 0) in chunk2

        // For vertex (0, 1) in chunk1, it should remain the same
        assert_eq!(chunk1_normals[2], Vec3::new(0.0, 1.0, 0.0)); // Vertex (0, 1) in chunk1

        // For vertex (2, 0) in chunk2, it should remain the same
        assert_eq!(chunk2_normals[1], Vec3::new(0.0, 1.0, 0.0)); // Vertex (2, 0) in chunk2
    }
}
