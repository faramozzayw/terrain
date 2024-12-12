use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy::render::render_asset::RenderAssetUsages;
use bevy_flycam::PlayerPlugin;
use bevy_inspector_egui::quick::WorldInspectorPlugin;
use nalgebra::Vector3;
use rayon::prelude::*;

fn generate_plane_mesh(heightmap: Vec<Vec<f32>>, cell_size: f32, max_height: f32) -> Mesh {
    let rows = heightmap.len();
    let cols = heightmap[0].len();
    let size = rows * cols;
    let mut positions = Vec::with_capacity(size);
    let mut uvs = Vec::with_capacity(size);
    let mut indices = Vec::with_capacity((rows - 1) * (cols - 1) * 6);
    let mut normals = vec![Vector3::default(); size];

    let inv_cols_minus_1 = 1.0 / (cols - 1) as f32;
    let inv_rows_minus_1 = 1.0 / (rows - 1) as f32;

    for (i, row) in heightmap.iter().enumerate() {
        for (j, column) in row.iter().enumerate() {
            let height = column * max_height;
            positions.push(Vector3::new(
                j as f32 * cell_size,
                height,
                i as f32 * cell_size,
            ));
            uvs.push([j as f32 * inv_cols_minus_1, i as f32 * inv_rows_minus_1]);
        }
    }

    let row_offsets = (0..rows).map(|i| i * cols).collect::<Vec<usize>>();

    for i in 0..rows - 1 {
        for j in 0..cols - 1 {
            let i0 = row_offsets[i] + j;
            let i1 = i0 + 1;
            let i2 = row_offsets[i + 1] + j;
            let i3 = i2 + 1;

            let normal1 = calculate_normal(positions[i0], positions[i2], positions[i1]);
            let normal2 = calculate_normal(positions[i1], positions[i2], positions[i3]);

            indices.extend_from_slice(&[
                i0 as u32, i2 as u32, i1 as u32, i1 as u32, i2 as u32, i3 as u32,
            ]);

            normals[i0] += normal1;
            normals[i2] += normal1;
            normals[i1] += normal1;

            normals[i1] += normal2;
            normals[i2] += normal2;
            normals[i3] += normal2;
        }
    }

    normals
        .par_iter_mut()
        .for_each(|normal| *normal = normal.normalize());

    let positions = positions
        .into_par_iter()
        .map(Into::into)
        .collect::<Vec<[f32; 3]>>();
    let normals = normals
        .into_par_iter()
        .map(Into::into)
        .collect::<Vec<[f32; 3]>>();

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));

    mesh
}

#[inline]
fn calculate_normal(
    vertex_0: Vector3<f32>,
    vertex_1: Vector3<f32>,
    vertex_2: Vector3<f32>,
) -> Vector3<f32> {
    let u = vertex_1 - vertex_0;
    let v = vertex_2 - vertex_0;

    u.cross(&v)
}

fn parse_heightmap(path: &str) -> Vec<Vec<f32>> {
    let img = image::open(path)
        .expect("Failed to open heightmap")
        .into_luma8(); // Convert to grayscale

    let (width, height) = img.dimensions();
    let mut heightmap = Vec::with_capacity(height as usize);

    for y in 0..height {
        let mut row = Vec::with_capacity(width as usize);
        for x in 0..width {
            let pixel_value = img.get_pixel(x, y)[0]; // Grayscale intensity
            row.push(pixel_value as f32 / 255.0); // Normalize to [0.0, 1.0]
        }
        heightmap.push(row);
    }

    heightmap
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(PlayerPlugin)
        .add_plugins(WorldInspectorPlugin::new())
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let heightmap = parse_heightmap("assets/heightmap.png");
    let plane_mesh = generate_plane_mesh(heightmap, 1.0, 100.0);

    commands.spawn(PbrBundle {
        mesh: meshes.add(plane_mesh),
        material: materials.add(Color::srgb(0.2, 0.8, 0.2)),
        transform: Transform::from_xyz(0.0, 0.0, 0.0).with_scale(Vec3::splat(0.005)),
        ..Default::default()
    });

    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 10_000.0,
            ..default()
        },
        transform: Transform::from_xyz(5.0, 5.0, 5.0).with_rotation(Quat::from_euler(
            EulerRot::XYZ,
            -0.1,
            0.0,
            0.0,
        )),
        ..default()
    });
}
