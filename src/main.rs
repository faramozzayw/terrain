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
        .add_systems(Update, create_array_texture)
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
    #[texture(100, dimension = "2d_array")]
    #[sampler(101)]
    array_texture: Handle<Image>,

    #[texture(102, dimension = "2d", sample_type = "u_int")]
    #[sampler(103, sampler_type = "non_filtering")]
    material_index_map: Handle<Image>,
}

impl MaterialExtension for TerrainExtension {
    fn fragment_shader() -> render_resource::ShaderRef {
        "shaders/terrain_shader.wgsl".into()
    }

    fn deferred_fragment_shader() -> render_resource::ShaderRef {
        "shaders/terrain_shader.wgsl".into()
    }
}

#[derive(Resource)]
struct LoadingTexture {
    is_loaded: bool,
    array_texture: Handle<Image>,
    material_index_map: Handle<Image>,
}

fn create_array_texture(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut loading_texture: ResMut<LoadingTexture>,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut terrain_materials: ResMut<Assets<ExtendedMaterial<StandardMaterial, TerrainExtension>>>,
) {
    let is_textures_are_loaded = asset_server
        .load_state(&loading_texture.material_index_map)
        .is_loaded()
        && asset_server
            .load_state(&loading_texture.array_texture)
            .is_loaded();

    if loading_texture.is_loaded || !is_textures_are_loaded {
        return;
    }
    loading_texture.is_loaded = true;

    if let Some(image) = images.get_mut(&loading_texture.array_texture) {
        let array_layers = 5;
        image.reinterpret_stacked_2d_as_array(array_layers);
    }

    if let Some(image) = images.get_mut(&loading_texture.material_index_map) {
        image.texture_descriptor.format = render_resource::TextureFormat::Rgba8Uint;
        image.sampler = bevy::image::ImageSampler::nearest();
    }

    let extension_handle = terrain_materials.add(ExtendedMaterial {
        base: StandardMaterial {
            base_color: Color::NONE,
            ..Default::default()
        },
        extension: TerrainExtension {
            array_texture: loading_texture.array_texture.clone(),
            material_index_map: loading_texture.material_index_map.clone(),
        },
    });

    let heightmap = parse_heightmap("assets/heightmap2.png");
    let num_chunks = heightmap.len() / Chunk::CHUNK_SIZE;

    let terrain = Terrain::generate_chunks(&heightmap, num_chunks);

    for chunk in &terrain.chunks {
        let x = (chunk.chunk_x * Chunk::CHUNK_SIZE) as f32;
        let z = (chunk.chunk_y * Chunk::CHUNK_SIZE) as f32;

        commands.spawn((
            Name::new(format!("Chunk [{x},{z}]")),
            Mesh3d(meshes.add(chunk.clone().into_mesh())),
            MeshMaterial3d(extension_handle.clone()),
            Transform::from_xyz(x, 0.0, z),
        ));
    }
}

fn setup(
    asset_server: Res<AssetServer>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut standard_materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.insert_resource(LoadingTexture {
        is_loaded: false,
        array_texture: asset_server.load("textures/array_texture.png"),
        material_index_map: asset_server.load("textures/custom_map.png"),
    });

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
