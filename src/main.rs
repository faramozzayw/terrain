mod terrain;
mod utils;

use bevy::{color::palettes::tailwind::YELLOW_400, prelude::*};
use bevy_flycam::{MovementSettings, PlayerPlugin};
use bevy_inspector_egui::quick::WorldInspectorPlugin;
use terrain::{Chunk, Terrain};
use utils::parse_heightmap;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(PlayerPlugin)
        .add_plugins(WorldInspectorPlugin::new())
        .add_systems(Startup, setup)
        .insert_resource(MovementSettings {
            sensitivity: 0.00015,
            speed: 60.0,
        })
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let heightmap = parse_heightmap("assets/heightmap2.png");
    let num_chunks = heightmap.len() / Chunk::CHUNK_SIZE;

    let terrain = Terrain::generate_chunks(&heightmap, num_chunks);

    for chunk in &terrain.chunks {
        let x = (chunk.chunk_x * Chunk::CHUNK_SIZE) as f32;
        let z = (chunk.chunk_y * Chunk::CHUNK_SIZE) as f32;

        commands.spawn((
            Name::new(format!("Chunk [{x},{z}]")),
            PbrBundle {
                mesh: meshes.add(chunk.clone().into_mesh()),
                material: materials.add(Color::srgb(0.2, 0.8, 0.2)),
                transform: Transform::from_xyz(x, 0.0, z),
                ..Default::default()
            },
        ));
    }

    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 10_000.0,
            color: YELLOW_400.into(),
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
