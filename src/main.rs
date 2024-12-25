mod terrain;
mod utils;

use bevy::{
    color::palettes::css::WHITE,
    pbr::{ExtendedMaterial, MaterialExtension},
    prelude::*,
    render::render_resource::{self, AsBindGroup},
};
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
        .add_plugins(MaterialPlugin::<
            ExtendedMaterial<StandardMaterial, TerrainExtension>,
        >::default())
        .run();
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct TerrainExtension {
    #[texture(100)]
    #[sampler(101)]
    pub grass_texture: Handle<Image>,

    #[texture(102)]
    #[sampler(103)]
    pub rock_texture: Handle<Image>,

    #[texture(104)]
    #[sampler(105)]
    pub snow_texture: Handle<Image>,
}

impl MaterialExtension for TerrainExtension {
    fn fragment_shader() -> render_resource::ShaderRef {
        "shaders/terrain_shader.wgsl".into()
    }

    fn deferred_fragment_shader() -> render_resource::ShaderRef {
        "shaders/terrain_shader.wgsl".into()
    }
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut terrain_materials: ResMut<Assets<ExtendedMaterial<StandardMaterial, TerrainExtension>>>,
    mut standard_materials: ResMut<Assets<StandardMaterial>>,
) {
    let heightmap = parse_heightmap("assets/heightmap2.png");
    let num_chunks = heightmap.len() / Chunk::CHUNK_SIZE;

    let terrain = Terrain::generate_chunks(&heightmap, num_chunks);

    for chunk in &terrain.chunks {
        let x = (chunk.chunk_x * Chunk::CHUNK_SIZE) as f32;
        let z = (chunk.chunk_y * Chunk::CHUNK_SIZE) as f32;

        commands.spawn((
            Name::new(format!("Chunk [{x},{z}]")),
            Mesh3d(meshes.add(chunk.clone().into_mesh())),
            MeshMaterial3d(terrain_materials.add(ExtendedMaterial {
                base: StandardMaterial {
                    base_color: Color::NONE,
                    ..Default::default()
                },
                extension: TerrainExtension {
                    grass_texture: asset_server.load("textures/grass.jpg"),
                    rock_texture: asset_server.load("textures/rock.jpg"),
                    snow_texture: asset_server.load("textures/snow.jpg"),
                },
            })),
            Transform::from_xyz(x, 0.0, z),
        ));
    }
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(1.0, 1.0, 1.0))),
        MeshMaterial3d(standard_materials.add(Color::srgb_u8(124, 144, 255))),
        Transform::from_xyz(0.0, 0.5, 0.0),
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 10_000.0,
            color: WHITE.into(),
            ..default()
        },
        Transform::from_xyz(5.0, 5.0, 5.0).with_rotation(Quat::from_euler(
            EulerRot::XYZ,
            -0.1,
            0.0,
            0.0,
        )),
    ));
}
