pub mod brushes;
mod terrain;
mod utils;

use std::ops::DerefMut;

use bevy::{
    color::palettes::css::WHITE,
    pbr::{ExtendedMaterial, MaterialExtension},
    picking::{backend::ray::RayMap, mesh_picking::ray_cast::RayMeshHit},
    prelude::*,
    render::render_resource::{self, AsBindGroup},
};
use bevy_flycam::{MovementSettings, PlayerPlugin};
use bevy_inspector_egui::quick::WorldInspectorPlugin;
use brushes::{apply_brush, BrushConfig, BrushKind};
use itertools::Itertools;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use terrain::{Chunk, Terrain};
use utils::parse_heightmap;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(PlayerPlugin)
        .add_plugins(WorldInspectorPlugin::new())
        .add_systems(Startup, setup)
        .add_event::<RecalculateNormals>()
        .add_systems(
            Update,
            (
                handle_keyboard_input,
                spawn_terrain,
                raycast.pipe(raycast_handle).pipe(pipe_noop_option),
            ),
        )
        .add_systems(Update, recalculate_normals_for_chunk.pipe(pipe_noop_option))
        .insert_resource(BrushConfig::default())
        .register_type::<BrushConfig>()
        .register_type::<ChunkMetadata>()
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
    #[uniform(100)]
    layers: u32,

    #[uniform(101)]
    tiling_factor: f32,

    #[texture(102, dimension = "2d_array")]
    #[sampler(103)]
    array_texture: Handle<Image>,

    #[texture(104, dimension = "2d", sample_type = "u_int")]
    #[sampler(105, sampler_type = "non_filtering")]
    material_index_map: Handle<Image>,

    #[texture(106, dimension = "2d_array")]
    #[sampler(107)]
    array_normal: Handle<Image>,

    #[texture(108, dimension = "2d_array")]
    #[sampler(109)]
    array_roughness: Handle<Image>,
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
struct LoadingTerrainTexture {
    is_loaded: bool,
    array_texture: Handle<Image>,
    array_normal: Handle<Image>,
    array_roughness: Handle<Image>,

    material_index_map: Handle<Image>,
}

impl LoadingTerrainTexture {
    pub fn is_textures_are_loaded(&self, asset_server: &AssetServer) -> bool {
        let is_mat_map_loaded = asset_server
            .load_state(&self.material_index_map)
            .is_loaded();
        let is_array_texture_loaded = asset_server.load_state(&self.array_texture).is_loaded();
        let is_array_normal_loaded = asset_server.load_state(&self.array_normal).is_loaded();
        let is_array_roughness_loaded = asset_server.load_state(&self.array_roughness).is_loaded();

        is_mat_map_loaded
            && is_array_texture_loaded
            && is_array_normal_loaded
            && is_array_roughness_loaded
    }
}

#[derive(Debug, Component, PartialEq, Eq, Reflect)]
#[reflect(Component)]
pub struct ChunkMetadata {
    x: usize,
    y: usize,
}

impl ChunkMetadata {
    pub fn new(x: usize, y: usize) -> Self {
        Self { x, y }
    }

    pub fn vec2(&self) -> Vec2 {
        Vec2::new(self.x as f32, self.y as f32)
    }
}

fn spawn_terrain(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut loading_texture: ResMut<LoadingTerrainTexture>,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut terrain_materials: ResMut<Assets<ExtendedMaterial<StandardMaterial, TerrainExtension>>>,
) {
    if loading_texture.is_loaded || !loading_texture.is_textures_are_loaded(&asset_server) {
        return;
    }
    loading_texture.is_loaded = true;
    let mut layers = 0;

    if let Some(image) = images.get_mut(&loading_texture.array_texture) {
        layers = image.height() / image.width();
        image.reinterpret_stacked_2d_as_array(layers);
    }

    if let Some(image) = images.get_mut(&loading_texture.array_normal) {
        assert_eq!(image.height() / image.width(), layers);
        image.reinterpret_stacked_2d_as_array(layers);
    }

    if let Some(image) = images.get_mut(&loading_texture.array_roughness) {
        assert_eq!(image.height() / image.width(), layers);
        image.reinterpret_stacked_2d_as_array(layers);
    }

    if let Some(image) = images.get_mut(&loading_texture.material_index_map) {
        image.texture_descriptor.format = render_resource::TextureFormat::Rgba8Uint;
        image.sampler = bevy::image::ImageSampler::nearest();
    }

    let extension_handle = terrain_materials.add(ExtendedMaterial {
        base: StandardMaterial {
            base_color: Color::NONE,
            // metallic: 0.0,
            // reflectance: 0.0,
            ..Default::default()
        },
        extension: TerrainExtension {
            array_texture: loading_texture.array_texture.clone(),
            array_normal: loading_texture.array_normal.clone(),
            array_roughness: loading_texture.array_roughness.clone(),
            material_index_map: loading_texture.material_index_map.clone(),
            layers,
            tiling_factor: 10.0,
        },
    });

    let heightmap = parse_heightmap("assets/heightmap2.png");
    let num_chunks = heightmap.len() / Chunk::SIZE;

    let terrain = Terrain::generate_chunks(&heightmap, num_chunks);

    let chunks = terrain
        .chunks
        .into_par_iter()
        .map(|chunk| (chunk.chunk_x, chunk.chunk_y, chunk.into_mesh()))
        .collect::<Vec<_>>();

    for (chunk_x, chunk_y, mesh) in chunks {
        commands.spawn((
            Name::new(format!("Chunk {}x{}", chunk_x, chunk_y)),
            Mesh3d(meshes.add(mesh)),
            ChunkMetadata::new(chunk_x, chunk_y),
            MeshMaterial3d(extension_handle.clone()),
            Transform::from_xyz(
                (chunk_x * Chunk::SIZE) as f32,
                0.0,
                (chunk_y * Chunk::SIZE) as f32,
            ),
        ));
    }
}

fn raycast(
    mut ray_cast: MeshRayCast,
    ray_map: Res<RayMap>,
    keys: Res<ButtonInput<MouseButton>>,
    mut config: ResMut<BrushConfig>,
    mut gizmos: Gizmos,
) -> Option<(Entity, RayMeshHit)> {
    for (_, ray) in ray_map.iter() {
        let Some((entity, hit)) = ray_cast.cast_ray(*ray, &RayCastSettings::default()).first()
        else {
            continue;
        };

        gizmos
            .circle(
                Isometry3d::new(hit.point, Quat::from_rotation_arc(Vec3::Z, Vec3::Y)),
                config.range,
                bevy::color::palettes::css::NAVY,
            )
            .resolution(64);

        if keys.pressed(MouseButton::Right) && matches!(config.kind, BrushKind::Sculp) {
            config.should_inverse_strength = true;
        } else if keys.pressed(MouseButton::Left) && matches!(config.kind, BrushKind::Sculp) {
            config.should_inverse_strength = false;
        } else if !keys.pressed(MouseButton::Left) {
            config.flatten_level = None;
            return None;
        }

        return Some((*entity, hit.clone()));
    }

    None
}

fn raycast_handle(
    In(maybe_hit): In<Option<(Entity, RayMeshHit)>>,
    mesh_3d: Query<(&GlobalTransform, &Mesh3d)>,
    metadata: Query<(Entity, &ChunkMetadata)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut config: ResMut<BrushConfig>,
    mut recalculate_normals: EventWriter<RecalculateNormals>,
) -> Option<()> {
    let (entity, hit) = maybe_hit?;
    let (chunk_gt, _) = mesh_3d.get(entity).ok()?;
    let center_position = chunk_gt.affine().inverse().transform_point3(hit.point);
    let center_vec2 = hit.point.xz();

    if matches!(config.kind, BrushKind::Flatten) && config.flatten_level.is_none() {
        config.flatten_level = Some(center_position.y);
    }

    let grid_size = (metadata.iter().count() as f32).sqrt() as usize;
    let chunk_size_f32 = Chunk::SIZE as f32;
    let half_square = chunk_size_f32 / 2.0;
    let half_diagonal = (half_square * 2.0_f32.sqrt()).ceil();
    let radius = config.range;

    let entities = (0..grid_size)
        .flat_map(|ref i| {
            (0..grid_size)
                .filter_map(|j| {
                    let square_center = (Vec2::new(*i as f32, j as f32) + 0.5) * chunk_size_f32;
                    let distance = (center_vec2 - square_center).length_squared().sqrt();

                    if distance <= radius + half_diagonal {
                        metadata.iter().find_map(|(e, metadata)| {
                            ChunkMetadata::new(*i, j).eq(metadata).then_some(e)
                        })
                    } else {
                        None
                    }
                })
                .collect_vec()
        })
        .collect_vec();

    let meshes_ptr: *mut Assets<Mesh> = meshes.deref_mut();

    entities
        .iter()
        .filter_map(|entity| {
            let (chunk_gt, chunk) = mesh_3d.get(*entity).ok()?;
            let center = chunk_gt.affine().inverse().transform_point3(hit.point).xz();

            // SAFETY: Each chunk mesh is unique to each chunk entity
            let chunk_mesh = unsafe {
                let meshes = &mut *meshes_ptr;
                meshes.get_mut(&chunk.0)?
            };

            Some((center, chunk_mesh))
        })
        .collect_vec()
        .into_par_iter()
        .map(|(center, chunk_mesh)| apply_brush(center, chunk_mesh, &config))
        .collect::<Option<Vec<_>>>()?;

    recalculate_normals.send_batch(entities.into_iter().map(RecalculateNormals));

    Some(())
}

#[derive(Debug, Event, PartialEq, Eq, Hash, Deref)]
pub struct RecalculateNormals(Entity);

// TODO: this is kinda fucked up, but fine for now
fn recalculate_normals_for_chunk(
    mut recalculate_normals: EventReader<RecalculateNormals>,
    mesh_3d: Query<&Mesh3d>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut local_timer: Local<Timer>,
    time: Res<Time>,
) -> Option<()> {
    if local_timer.duration().as_nanos() == 0 {
        local_timer.set_duration(std::time::Duration::from_secs_f32(0.05));
        local_timer.set_mode(TimerMode::Repeating);
        return Some(());
    }
    if local_timer.finished() {
        local_timer.reset();
    } else {
        local_timer.tick(time.delta());
        return Some(());
    }

    let chunks_entities = recalculate_normals
        .read()
        .unique()
        .map(std::ops::Deref::deref)
        .collect_vec();

    for entity in chunks_entities {
        let chunk = mesh_3d.get(*entity).ok()?;
        let chunk_mesh = meshes.get_mut(&chunk.0)?;
        chunk_mesh.compute_normals();
    }

    Some(())
}

fn handle_keyboard_input(
    mut config: ResMut<BrushConfig>,
    mut light: Query<&mut DirectionalLight>,
    keys: Res<ButtonInput<KeyCode>>,
) {
    if keys.just_pressed(KeyCode::Digit1) {
        config.kind = BrushKind::Sculp;
    }

    if keys.just_pressed(KeyCode::Digit2) {
        config.kind = BrushKind::Flatten;
    }

    if keys.just_pressed(KeyCode::Digit3) {
        config.kind = BrushKind::Smooth;
    }

    if keys.pressed(KeyCode::PageUp) {
        config.range += 0.25;
    }

    if keys.pressed(KeyCode::PageDown) {
        config.range -= 0.25;
    }

    if keys.just_pressed(KeyCode::KeyL) {
        let mut light = light.single_mut();
        light.shadows_enabled = !light.shadows_enabled;
    }
}

fn pipe_noop_option<T>(_: In<Option<T>>) {}

fn setup(
    asset_server: Res<AssetServer>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut config_store: ResMut<GizmoConfigStore>,
    mut standard_materials: ResMut<Assets<StandardMaterial>>,
) {
    for (_, config, _) in config_store.iter_mut() {
        config.depth_bias = -1.;
    }

    commands.insert_resource(LoadingTerrainTexture {
        is_loaded: false,
        array_normal: asset_server.load("textures/array_normal.png"),
        array_texture: asset_server.load("textures/array_texture.png"),
        array_roughness: asset_server.load("textures/array_roughness.png"),
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
            0.1,
            3.6,
            0.0,
        )),
    ));
}
